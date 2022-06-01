# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import bernoulli

"""## RL Algorithms
---
"""

## CLIPPING VALUES ##
MIN_CLIP_VALUE = 0.35
MAX_CLIP_VALUE = 0.75
# Advantage Time Feature Dimensions
D_advantage = 3
# Baseline Time Feature Dimensions
D_baseline = 4
# Number of Posterior Draws
NUM_POSTERIOR_SAMPLES = 5000

## HELPERS ##
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def process_alg_state(env_state, env_type):
    if env_type == 'stat':
        baseline_state = np.array([env_state[0], env_state[1], env_state[3], 1])
        advantage_state = np.delete(baseline_state, 2)
    else:
        baseline_state = np.array([env_state[0], env_state[1], env_state[4], 1])
        advantage_state = np.delete(baseline_state, 2)

    return advantage_state, baseline_state

class RLAlgorithmCandidate():
    def __init__(self, alg_type, cluster_size, update_cadence):
        self.alg_type = alg_type
        self.cluster_size = cluster_size
        self.update_cadence = update_cadence
        # process_alg_state is a global function
        self.process_alg_state_func = process_alg_state

    def action_selection(self, advantage_state, baseline_state):
        return 0

    def update(self, advantage_states, baseline_states, actions, pis, rewards):
        return 0

    def get_cluster_size(self):
        return self.cluster_size

    def get_update_cadence(self):
        return self.update_cadence

"""### Bayesian Linear Regression Thompson Sampler
---

#### Helper Functions
---
"""

## POSTERIOR HELPERS ##
# create the feature vector given state, action, and action selection probability
def create_big_phi(advantage_states, baseline_states, actions, probs):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, probs).T, \
                       np.multiply(advantage_states.T, (actions - probs)).T,))
  return big_phi

def compute_posterior_var(Phi, sigma_n_squared, prior_sigma):
  return np.linalg.inv(1/sigma_n_squared * Phi.T @ Phi + np.linalg.inv(prior_sigma))

def compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma):
  # return np.linalg.inv(1/sigma_n_squared * X.T @ X + np.linalg.inv(prior_sigma)) \
  # @ (1/sigma_n_squared * X.T @ y + (prior_mu @ np.linalg.inv(prior_sigma)).T)
  return compute_posterior_var(Phi, sigma_n_squared, prior_sigma) \
   @ (1/sigma_n_squared * Phi.T @ R + np.linalg.inv(prior_sigma) @ prior_mu)

# update posterior distribution
def update_posterior_w(Phi, R, sigma_n_squared, prior_mu, prior_sigma):
  mean = compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma)
  var = compute_posterior_var(Phi, sigma_n_squared, prior_sigma)

  return mean, var

def get_beta_posterior_draws(posterior_mean, posterior_var):
  # grab last D_advantage of mean vector
  beta_post_mean = posterior_mean[-D_advantage:]
  # grab right bottom corner D_advantage x D_advantage submatrix
  beta_post_var = posterior_var[-D_advantage:,-D_advantage:]

  return np.random.multivariate_normal(beta_post_mean, beta_post_var, NUM_POSTERIOR_SAMPLES)

## ACTION SELECTION ##
# we calculate the posterior probability of P(R_1 > R_0) clipped
# we make a Bernoulli draw with prob. P(R_1 > R_0) of the action
def bayes_lr_action_selector(beta_posterior_draws, advantage_state):
  num_positive_preds = len(np.where(beta_posterior_draws @ advantage_state > 0)[0])
  posterior_prob =  num_positive_preds / len(beta_posterior_draws)
  clipped_prob = max(min(MAX_CLIP_VALUE, posterior_prob), MIN_CLIP_VALUE)
  return bernoulli.rvs(clipped_prob), clipped_prob

""" #### BLR Algorithm Object
---
"""

class BayesianLinearRegression(RLAlgorithmCandidate):
    def __init__(self, cluster_size, update_cadence):
        super(BayesianLinearRegression, self).__init__('blr', cluster_size, update_cadence)

        self.PRIOR_MU = np.zeros(D_baseline + D_advantage + D_advantage)
        self.PRIOR_SIGMA = 5 * np.eye(len(self.PRIOR_MU))
        self.SIGMA_N_2 = 3526.747
        # initial draws are from the prior
        self.beta_posterior_draws = get_beta_posterior_draws(self.PRIOR_MU, self.PRIOR_SIGMA)

    def action_selection(self, advantage_state, baseline_state):
        return bayes_lr_action_selector(self.beta_posterior_draws, advantage_state)

    def update(self, advantage_states, baseline_states, actions, pis, rewards):
        Phi = create_big_phi(advantage_states, baseline_states, actions, pis)
        posterior_mean, posterior_var = update_posterior_w(Phi, rewards, self.SIGMA_N_2, self.PRIOR_MU, self.PRIOR_SIGMA)
        self.beta_posterior_draws = get_beta_posterior_draws(posterior_mean, posterior_var)

"""### Zero-Inflated Poisson with Metropolis-Hasting (MH) Sampler
---
"""

from scipy.stats import uniform
from scipy.stats import multivariate_normal
import math

def prior_density(w_b_0, w_p_0, w_b_1, w_p_1):
    log_prior_w_b_0_density = multivariate_normal.logpdf(w_b_0, mean=np.zeros(D_baseline), cov=np.ones(D_baseline))
    log_prior_w_b_1_density = multivariate_normal.logpdf(w_b_1, mean=np.zeros(D_advantage), cov=np.ones(D_advantage))
    log_prior_w_p_0_density = multivariate_normal.logpdf(w_p_0, mean=np.zeros(D_baseline), cov=np.ones(D_baseline))
    log_prior_w_p_1_density = multivariate_normal.logpdf(w_p_1, mean=np.zeros(D_advantage), cov=np.ones(D_advantage))

    return log_prior_w_b_0_density + log_prior_w_b_1_density + \
    log_prior_w_p_0_density + log_prior_w_p_1_density

## Potential Problem: What if the reward is not an integer?
def llkhd_density(advantage_state, baseline_state, action, \
                  w_b_0, w_p_0, w_b_1, w_p_1, obs):
    bern_p = 1 - sigmoid(baseline_state @ w_b_0 + \
                         action * advantage_state @ w_b_1)
    x = baseline_state @ w_p_0 + action * advantage_state @ w_p_1
    lam = np.exp(x)
    # ref: https://discourse.pymc.io/t/zero-inflated-poisson-log-lik/2664
    # density of a 0-inflated poisson
    if obs == 0:
      return np.log((1 - bern_p) + bern_p * np.exp(-lam))
    else:
      return np.log(bern_p) + (-lam) + obs * np.log(lam) - math.log(np.math.factorial(int(obs)))

def log_posterior_density(advantage_states, baseline_states, actions, \
                          w_b_0, w_p_0, w_b_1, w_p_1, obs):
    log_prior = prior_density(w_b_0, w_p_0, w_b_1, w_p_1)
    log_llkhd = np.sum([llkhd_density(advantage_states[i], baseline_states[i], actions[i], \
                                      w_b_0, w_p_0, w_b_1, w_p_1, obs[i]) for i in range(len(advantage_states))])

    return log_prior + log_llkhd

## we want an acceptance rate of 25-50% for MH
## if not, tune step_size
# ref for choosing step size: https://stackoverflow.com/questions/28686900/how-to-decide-the-step-size-when-using-metropolis-hastings-algorithm
def metropolis_hastings(advantage_states, baseline_states, actions, Y, \
                        step_size=0.01, num_steps=10000):
  w_b_0_old, w_p_0_old = np.random.randn(D_baseline) * 0.5, np.random.randn(D_baseline) * 0.5
  w_b_1_old, w_p_1_old = np.random.randn(D_advantage) * 0.5, np.random.randn(D_advantage) * 0.5
  log_post_dist = lambda w_b_0, w_p_0, w_b_1, w_p_1 : log_posterior_density(advantage_states, baseline_states, actions, \
                                                        w_b_0, w_p_0, w_b_1, w_p_1, Y)
  old_logp_val = log_post_dist(w_b_0_old, w_p_0_old, w_b_1_old, w_p_1_old)
  accepted_samples = []
  num_accepts = 0
  for step in range(num_steps):
    if step > 0 and step % 1000 == 0:
      print("ITERATION: {}, Acceptance Rate for MH is {}".format(step, num_accepts / step))
    # propose a new sample
    w_b_0_prop = multivariate_normal(mean=w_b_0_old, cov=step_size**2 * np.eye(D_baseline)).rvs()
    w_p_0_prop = multivariate_normal(mean=w_p_0_old, cov=step_size**2 * np.eye(D_baseline)).rvs()
    w_b_1_prop = multivariate_normal(mean=w_b_1_old, cov=step_size**2 * np.eye(D_advantage)).rvs()
    w_p_1_prop = multivariate_normal(mean=w_p_1_old, cov=step_size**2 * np.eye(D_advantage)).rvs()
    # accept or reject
    U = uniform.rvs()
    prop_logp_val = log_post_dist(w_b_0_prop, w_p_0_prop, w_b_1_prop, w_p_1_prop)
    accept_prob = prop_logp_val - old_logp_val
    if np.log(U) < accept_prob:
      accepted_samples.append(np.concatenate((w_b_0_prop, w_p_0_prop, w_b_1_prop, w_p_1_prop), axis=0))
      old_logp_val = prop_logp_val
      w_b_0_old = w_b_0_prop
      w_p_0_old = w_p_0_prop
      w_b_1_old = w_b_1_prop
      w_p_1_old = w_p_1_prop
      num_accepts += 1
    else:
      accepted_samples.append(np.concatenate((w_b_0_old, w_p_0_old, w_b_1_old, w_p_1_old), axis=0))

  return accepted_samples

# defined burn-in
BURN_IN = .1
# define thinning
THIN = 2

def burn_and_thin(samples):
  N = len(samples)
  burn_thin_samples = samples[int(BURN_IN * N)::THIN]

  return burn_thin_samples

def bayes_zero_inflated_pred(advantage_state, baseline_state, sample, action):
    w_b_0, w_p_0 = sample[:D_baseline], sample[D_baseline:2 * D_baseline]
    w_b_1, w_p_1 = sample[2 * D_baseline:2 * D_baseline + D_advantage], sample[2 * D_baseline + D_advantage:]
    bern_p = 1 - sigmoid(baseline_state @ w_b_0 + action * advantage_state @ w_b_1)
    # poisson component
    x = baseline_state @ w_p_0 + action * advantage_state @ w_p_1
    poisson_lam = np.exp(x)

    return bern_p * poisson_lam

def bayes_zero_inflated_action_selector(advantage_state, baseline_state, samples):
    posterior_preds_0 = np.apply_along_axis(lambda sample: bayes_zero_inflated_pred(advantage_state, baseline_state, sample, 0), axis=1, arr=samples)
    posterior_preds_1 = np.apply_along_axis(lambda sample: bayes_zero_inflated_pred(advantage_state, baseline_state, sample, 1), axis=1, arr=samples)
    diff = posterior_preds_1 - posterior_preds_0
    posterior_prob = np.count_nonzero(diff > 0) / len(samples)
    clipped_prob = max(min(MAX_CLIP_VALUE, posterior_prob), MIN_CLIP_VALUE)

    return bernoulli.rvs(clipped_prob), clipped_prob

""" #### ZIP Algorithm Object
---
"""

class BayesianZeroInflatedPoisson(RLAlgorithmCandidate):
    def __init__(self, cluster_size, update_cadence):
        super(BayesianZeroInflatedPoisson, self).__init__('zero_infl', cluster_size, update_cadence)

        self.PRIOR_MU = np.zeros(2 * D_baseline + 2 * D_advantage)
        # ANNA TODO: may need to change this later to make it more uninformative, but remember to change it
        # in the ZIP prior density function too
        self.PRIOR_SIGMA = np.eye(len(self.PRIOR_MU))
        # initial draws are from the prior
        self.theta_posterior_draws = \
        np.array([multivariate_normal(mean=self.PRIOR_MU, cov=self.PRIOR_SIGMA).rvs() for i in range(NUM_POSTERIOR_SAMPLES)])

    def action_selection(self, advantage_state, baseline_state):
        return bayes_zero_inflated_action_selector(advantage_state, baseline_state, self.theta_posterior_draws)

    def update(self, advantage_states, baseline_states, actions, pis, rewards):
        mh_samples = metropolis_hastings(advantage_states, baseline_states, actions, \
                                             rewards, num_steps=int((THIN*NUM_POSTERIOR_SAMPLES) / (1 - BURN_IN)))
        self.theta_posterior_draws = burn_and_thin(mh_samples)
