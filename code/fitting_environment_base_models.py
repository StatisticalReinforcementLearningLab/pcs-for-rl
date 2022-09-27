# -*- coding: utf-8 -*-
"""

# Fitting Environment Base Models
---
We have the following environment models:
* Stationary vs. Non-Stationary
* Base Model:
  1. Hurdle Model with Normal Square Root Transform and Linear Mean
  2. Hurdle Model with Normal Log Transform and Linear Mean
  3. Zero-Inflated Poisson Model

## Base Model Specifications
---
### Hurdle Model with Square Root Transform
---
$$
Z \sim Bern(1 - \sigma(x^Tw_b))
$$
$$
Y \sim \mathcal{N}(x^Tw_{\mu}, \sigma_u^2)
$$
$$
R = Z * Y^2
$$

### Hurdle Model with Log Transform
---
$$
Z \sim Bern(1 - \sigma(x^Tw_b))
$$
$$
Y \sim \text{LogNormal}(x^Tw_{\mu}, \sigma_u^2)
$$
$$
R = Z * Y
$$

### 0-Inflated Poisson Model
---
$$
Z \sim Bern(\sigma(x^Tw_b))
$$
$$
Y \sim Poisson(\exp(x^Tw_p))
$$
$$
R = Z * Y
$$
"""

# pull packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import poisson
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pymc3 as pm
from pymc3.model import Model
import theano.tensor as tt
import arviz as az

import pickle

"""## Creating the State Space and Reward From ROBAS 2 Data
---
"""

# get all robas 2 users
df = pd.read_csv('https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.2/master/inst/extdata/robas_2_data.csv')
users_df = df[['User']].drop_duplicates()

# non-zero values of total brush times
robas_2_user_total_brush_times = np.array(df['Brush Time'])[::2] + np.array(df['Brush Time'])[1::2]
# non_zero_total_brush_times = robas_2_user_total_brush_times[np.nonzero(robas_2_user_total_brush_times)]
print("Empirical Mean: ", np.mean(robas_2_user_total_brush_times))
print("Empirical Std: ", np.std(robas_2_user_total_brush_times))

# try both and see which one does better
def min_max_normalization(time):
  return (time - np.min(robas_2_user_total_brush_times)) / np.max(robas_2_user_total_brush_times) - np.min(robas_2_user_total_brush_times)

## HELPER FUNCTIONS ##
# normalized values derived from ROBAS 2 dataset
# : = (time - empirical mean of non_zero_total_brush_times)
# / empirical std of non_zero_total_brush_times
# z-score normalization
def normalize_total_brush_time(time):
  # return (time - 84) / 77
  return (time - 172) / 118

# these values are dependent on that the ROBAS 2 study lasted for 28 days
# this would need to be recalculated for studies of different lengths
def normalize_day_in_study(day):
  return (day - 14.5) / 13.5

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# generating stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Prop. Non-Zero Brushing In Past 7 Days
# 3 - Weekday vs. Weekend
# 4 - Bias
def generate_state_spaces(user_df, rewards):
  ## init ##
  D = 5
  states = np.zeros(shape=(len(user_df), D))
  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[3]
    # prior day brush time at same time of day
    if i > 1:
      if states[i][0] == 0:
        states[i][1] = normalize_total_brush_time(rewards[i - 1] + rewards[i - 2])
      else:
        states[i][1] = normalize_total_brush_time(rewards[i - 2] + rewards[i - 3])
    # prop. brushed in past 7 days
    if i > 13:
      states[i][2] = df_array[6]
    # weekday or weekend term
    states[i][3] = df_array[5]
    # bias term
    states[i][4] = 1

  return states

# generating non-stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Day In Study
# 3 - Prop. Non-Zero Brushing In Past 7 Days
# 4 - Weekday vs. Weekend
# 5 - Bias
def generate_state_spaces_non_stationarity(user_df, rewards):
  ## init ##
  D = 6
  states = np.zeros(shape=(len(user_df), D))
  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[3]
    # prior day brush time at same time of day
    if i > 1:
      if states[i][0] == 0:
        states[i][1] = normalize_total_brush_time(rewards[i - 1] + rewards[i - 2])
      else:
        states[i][1] = normalize_total_brush_time(rewards[i - 2] + rewards[i - 3])
    # day in study
    states[i][2] = normalize_day_in_study(df_array[2])
    # prop. brushed in past 7 days
    if i > 13:
      states[i][3] = df_array[6]
    # weekday or weekend term
    states[i][4] = df_array[5]
    # bias term
    states[i][5] = 1

  return states

def get_rewards(user_id):
  return np.array(df[df['User'] == user_id]['Brush Time'])

def get_user_df(user_id):
  return df[df['User'] == user_id]

# generate N=32 users
NUM_USERS = len(users_df)
NUM_DAYS = 28
NUM_SESSIONS = 2 * NUM_DAYS

# dictionary where key is user id and values are lists of sessions of trial
users_sessions = {}
users_sessions_non_stationarity = {}
users_rewards = {}
for index, row in users_df.iterrows():
  user_id = row['User']
  user_rewards = get_rewards(user_id)
  users_rewards[user_id] = user_rewards
  user_df = get_user_df(user_id)
  users_sessions[user_id] = generate_state_spaces(user_df, user_rewards)
  users_sessions_non_stationarity[user_id] = generate_state_spaces_non_stationarity(user_df, user_rewards)

"""## Fitting the Models
---
"""

# ref: https://pymc3-testing.readthedocs.io/en/rtd-docs/api/distributions/discrete.html#pymc3.distributions.discrete.ZeroInflatedPoisson
# e.g. https://discourse.pymc.io/t/zero-inflated-poisson-example/3862
# https://bwengals.github.io/gps-with-non-normal-likelihoods-in-pymc3.html
def build_bern_model(X, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_b = pm.MvNormal('w_b', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    bern_term = X @ w_b
    # Y:= vector of 0s and 1s
    R = pm.Bernoulli('R', p=1 - sigmoid(bern_term), observed=Y)

  return model

def build_log_norm_model(X, Y):
  model = pm.Model()
  with model:
    d = X.shape[1]
    w_mu = pm.MvNormal('w_mu', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    sigma_u = pm.Normal('sigma_u', mu=0, sigma=1)
    mean_term = X @ w_mu
    # we square the link function as a reparameterization
    R = pm.Lognormal('R', mu=mean_term, sigma=sigma_u, observed=Y)

  return model

def build_sqrt_norm_model(X, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_mu = pm.MvNormal('w_mu', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    sigma_u = pm.Normal('sigma_u', mu=0, sigma=1)
    mean_term = X @ w_mu
    # we square the link function as a reparameterization
    R = pm.Normal('R', mu=mean_term, sigma=sigma_u, observed=Y**0.5)

  return model

def build_0_inflated_poisson_model(X, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    # w_b = pm.Flat('w_b', shape=(d,))
    # w_p = pm.Flat('w_p', shape=(d,))
    w_b = pm.MvNormal('w_b', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    w_p = pm.MvNormal('w_p', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    bern_term = X @ w_b
    poisson_term = X @ w_p
    R = pm.ZeroInflatedPoisson("likelihood", psi=1 - sigmoid(bern_term), theta=tt.exp(poisson_term), observed=Y)

  return model

def turn_to_bern_labels(Y):
  return np.array([1 if val > 0 else 0 for val in Y])

def turn_to_norm_state_and_labels(X, Y):
  idxs = np.where(Y != 0)
  return X[idxs], Y[Y != 0]

from numpy.core.multiarray import concatenate

def run_bern_map_for_users(users_sessions, users_rewards, d, num_restarts):
  model_params = {}
  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_rewards = users_rewards[user_id]
    user_bern_rewards = turn_to_bern_labels(user_rewards)
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, d))
    for seed in range(num_restarts):
      model = build_bern_model(user_states, user_bern_rewards)
      np.random.seed(seed + 100)
      # init_params = {'w_b': np.random.randn(d)}
      init_params = {'w_b': np.ones(d) * seed}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_b = map_estimate['w_b']
      logp_vals[seed] = model.logp(map_estimate)
      param_vals[seed] = w_b
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

def run_normal_transform_map_for_users(users_sessions, users_rewards, d, num_restarts, transform):
  model_params = {}
  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_rewards = users_rewards[user_id]
    transform_norm_states, transform_norm_rewards = turn_to_norm_state_and_labels(user_states, user_rewards)
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, d + 1))
    for seed in range(num_restarts):
      # sqrt transform
      if transform == "sqrt":
        model = build_sqrt_norm_model(transform_norm_states, transform_norm_rewards)
      else:
        # log transform
        model = build_log_norm_model(transform_norm_states, transform_norm_rewards)
      np.random.seed(seed)
      init_params = {'w_mu': np.random.randn(d), 'sigma_u': abs(np.random.randn()),}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_mu = map_estimate['w_mu']
      sigma_u = map_estimate['sigma_u']
      logp_vals[seed] = model.logp(map_estimate)
      param_vals[seed] = np.concatenate((w_mu, np.array([sigma_u])), axis=0)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

def run_zero_infl_map_for_users(users_sessions, users_rewards, d, num_restarts):
  model_params = {}

  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_rewards = users_rewards[user_id]
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, 2 * d))
    for seed in range(num_restarts):
      model = build_0_inflated_poisson_model(user_states, user_rewards)
      np.random.seed(seed)
      init_params = {'w_b': np.random.randn(d), 'w_p':  np.random.randn(d)}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_b = map_estimate['w_b']
      w_p = map_estimate['w_p']
      logp_vals[seed] = model.logp(map_estimate)
      param_vals[seed] = np.concatenate((w_b, w_p), axis=None)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

"""### Stationary Models
---
"""

# stationary model, bernoulli params
stat_bern_model_params = run_bern_map_for_users(users_sessions, users_rewards, d=5, num_restarts=5)

# stationary model, log normal params
stat_log_norm_model_params = run_normal_transform_map_for_users(users_sessions, users_rewards, d=5, num_restarts=5, transform="log")

# stationary model, sqrt normal params
stat_sqrt_norm_model_params = run_normal_transform_map_for_users(users_sessions, users_rewards, d=5, num_restarts=5, transform="sqrt")

# stationary model, zip params
stat_zip_model_params = run_zero_infl_map_for_users(users_sessions, users_rewards, d=5, num_restarts=5)

"""### Non-Stationarity Models
---
"""

# non-stationary model, bernoulli params
non_stat_bern_model_params = run_bern_map_for_users(users_sessions_non_stationarity, users_rewards, d=6, num_restarts=5)

# non-stationary model, log normal params
non_stat_log_norm_model_params = run_normal_transform_map_for_users(users_sessions_non_stationarity, users_rewards, d=6, num_restarts=5, transform="log")

# non-stationary model, sqrt normal params
non_stat_sqrt_norm_model_params = run_normal_transform_map_for_users(users_sessions_non_stationarity, users_rewards, d=6, num_restarts=5, transform="sqrt")

# non-stationary model, zip params
non_stat_zip_model_params = run_zero_infl_map_for_users(users_sessions_non_stationarity, users_rewards, d=6, num_restarts=5)

"""### Saving Parameter Values
---
"""

# model_columns must contain "User" as the indexing key
def create_normal_transform_df_from_params(model_columns, bern_model_params, normal_transform_params):
  df = pd.DataFrame(columns = model_columns)
  for user in bern_model_params.keys():
    bern_vals = bern_model_params[user]
    norm_transform_vals = normal_transform_params[user]
    values = np.concatenate((bern_vals, norm_transform_vals), axis=0)
    new_row = {}
    new_row['User'] = user
    for i in range(1, len(model_columns)):
      new_row[model_columns[i]] = values[i - 1]
    df = df.append(new_row, ignore_index=True)

  return df

def create_zip_df_from_params(model_columns, zip_model_params):
  df = pd.DataFrame(columns = model_columns)
  for user in zip_model_params.keys():
    values = zip_model_params[user]
    new_row = {}
    new_row['User'] = user
    for i in range(1, len(model_columns)):
      new_row[model_columns[i]] = values[i - 1]
    df = df.append(new_row, ignore_index=True)

  return df

stat_normal_transform_model_columns = ['User', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Mu', 'Prior.Day.Total.Brush.Time.norm.Mu', 'Proportion.Brushed.In.Past.7.Days.Mu', 'Day.Type.Mu', 'Intercept.Mu', \
                        'Sigma_u']

non_stat_normal_transform_model_columns = ['User', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Day.in.Study.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Mu', 'Prior.Day.Total.Brush.Time.norm.Mu', 'Day.in.Study.norm.Mu', 'Proportion.Brushed.In.Past.7.Days.Mu', 'Day.Type.Mu', 'Intercept.Mu',
                        'Sigma_u']

stat_zero_infl_model_columns = ['User', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Poisson', 'Prior.Day.Total.Brush.Time.norm.Poisson', 'Proportion.Brushed.In.Past.7.Days.Poisson', 'Day.Type.Poisson', 'Intercept.Poisson']

non_stat_zero_infl_model_columns = ['User', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Day.in.Study.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Poisson', 'Prior.Day.Total.Brush.Time.norm.Poisson', 'Day.in.Study.norm.Poisson', 'Proportion.Brushed.In.Past.7.Days.Poisson', 'Day.Type.Poisson', 'Intercept.Poisson']

stat_sqrt_norm_df = create_normal_transform_df_from_params(stat_normal_transform_model_columns, stat_bern_model_params, stat_sqrt_norm_model_params)

stat_log_norm_df = create_normal_transform_df_from_params(stat_normal_transform_model_columns, stat_bern_model_params, stat_log_norm_model_params)

stat_zip_df = create_zip_df_from_params(stat_zero_infl_model_columns, stat_zip_model_params)

stat_sqrt_norm_df.to_csv('stat_sqrt_norm_hurdle_model_params.csv')
stat_log_norm_df.to_csv('stat_log_norm_hurdle_model_params.csv')
stat_zip_df.to_csv('stat_zero_infl_pois_model_params.csv')

non_stat_sqrt_norm_df = create_normal_transform_df_from_params(non_stat_normal_transform_model_columns, non_stat_bern_model_params, non_stat_sqrt_norm_model_params)

non_stat_log_norm_df = create_normal_transform_df_from_params(non_stat_normal_transform_model_columns, non_stat_bern_model_params, non_stat_log_norm_model_params)

non_stat_zip_df = create_zip_df_from_params(non_stat_zero_infl_model_columns, non_stat_zip_model_params)

non_stat_sqrt_norm_df.to_csv('non_stat_sqrt_norm_hurdle_model_params.csv')
non_stat_log_norm_df.to_csv('non_stat_log_norm_hurdle_model_params.csv')
non_stat_zip_df.to_csv('non_stat_zero_infl_pois_model_params.csv')
