# -*- coding: utf-8 -*-

# pull packages
import pandas as pd
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm

import pickle

# get all robas 2 users
df = pd.read_csv('../sim_env_data/robas_2_data.csv')
ROBAS_2_USERS = np.array(df[['User']].drop_duplicates()).flatten()

def normalize_total_brush_time(time):
  return (time - 172) / 118

# these values are dependent on that the ROBAS 2 study lasted for 28 days
# this would need to be recalculated for studies of different lengths
def normalize_day_in_study(day):
  return (day - 35.5) / 34.5

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# generating stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Prop. Non-Zero Brushing In Past 7 Days
# 3 - Weekday vs. Weekend
# 4 - Bias
def generate_state_spaces(user_df, num_days):
  ## init ##
  D = 5
  states = np.zeros(shape=(2 * num_days, D))
  for i in range(len(states)):
    # time of day
    states[i][0] = i % 2
    # bias term
    states[i][4] = 1

  # reinput weekday vs. weekend
  first_weekend_idx = np.where(np.array(user_df['Day Type']) == 1)[0][0]
  for j in range(4):
    states[first_weekend_idx + j::14,3] = 1

  return states

# generating non-stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Day In Study
# 3 - Prop. Non-Zero Brushing In Past 7 Days
# 4 - Weekday vs. Weekend
# 5 - Bias
def generate_state_spaces_non_stationarity(user_df, num_days):
  ## init ##
  D = 6
  states = np.zeros(shape=(2 * num_days, D))
  for i in range(len(states)):
    # time of day
    states[i][0] = i % 2
    # day in study
    states[i][2] = normalize_day_in_study(i // 2 + 1)
    # bias term
    states[i][5] = 1

  # reinput weekday vs. weekend
  first_weekend_idx = np.where(np.array(user_df['Day Type']) == 1)[0][0]
  for j in range(4):
    states[first_weekend_idx + j::14,4] = 1

  return states

def get_user_df(user_id):
  return df[df['User'] == user_id]

### ENVIRONMENT AND ALGORITHM STATE SPACE FUNCTIONS ###
def get_previous_day_total_brush_time(rewards, time_of_day, j):
    if j > 1:
        if time_of_day == 0:
            return rewards[j - 1] + rewards[j - 2]
        else:
            return rewards[j - 2] + rewards[j - 3]
    # first day there is no brushing
    else:
        return 0

def stat_process_env_state(session, j, rewards):
    env_state = session.copy()
    # session type - either 0 or 1
    session_type = int(env_state[0])
    # update previous day total brush time
    previous_day_total_rewards = get_previous_day_total_brush_time(rewards, session[0], j)
    env_state[1] = normalize_total_brush_time(previous_day_total_rewards)
    # proportion of past success brushing
    if (j >= 14):
      env_state[2] = np.sum([rewards[j - k] > 0.0 for k in range(1, 15)]) / 14

    return env_state

def non_stat_process_env_state(session, j, rewards):
    env_state = session.copy()
    # session type - either 0 or 1
    session_type = int(env_state[0])
    # update previous day total brush time
    previous_day_total_rewards = get_previous_day_total_brush_time(rewards, session[0], j)
    env_state[1] = normalize_total_brush_time(previous_day_total_rewards)
    # proportion of past success brushing
    if (j >= 14):
      env_state[3] = np.sum([rewards[j - k] > 0.0 for k in range(1, 15)]) / 14

    return env_state

"""## Generate States
---
"""

# generate N=32 users
NUM_USERS = len(ROBAS_2_USERS)
NUM_DAYS = 70
# dictionary where key is index and value is user_id
USER_INDICES = {}

# dictionary where key is user id and values are lists of sessions of trial
USERS_SESSIONS = {}
USERS_SESSIONS_NON_STATIONARITY = {}
for i, user_id in enumerate(ROBAS_2_USERS):
  user_idx = i
  USER_INDICES[user_idx] = user_id
  user_df = get_user_df(user_id)
  USERS_SESSIONS[user_id] = generate_state_spaces(user_df, NUM_DAYS)
  USERS_SESSIONS_NON_STATIONARITY[user_id] = generate_state_spaces_non_stationarity(user_df, NUM_DAYS)

## Square Root Transform ##
STAT_SQRT_NORM_DF = pd.read_csv("../sim_env_data/stat_sqrt_norm_hurdle_model_params.csv")
NON_STAT_SQRT_NORM_DF = pd.read_csv("../sim_env_data/non_stat_sqrt_norm_hurdle_model_params.csv")
## Log Transform ##
STAT_LOG_NORM_DF = pd.read_csv("../sim_env_data/stat_log_norm_hurdle_model_params.csv")
NON_STAT_LOG_NORM_DF = pd.read_csv("../sim_env_data/non_stat_log_norm_hurdle_model_params.csv")
## Zero Inflated Poisson ##
STAT_ZERO_INFL_POIS_DF = pd.read_csv("../sim_env_data/stat_zero_infl_pois_model_params.csv")
NON_STAT_ZERO_INFL_POIS_DF = pd.read_csv("../sim_env_data/non_stat_zero_infl_pois_model_params.csv")

def get_norm_transform_params_for_user(user, df, env_type='stat'):
  param_dim = 5 if env_type == 'stat' else 6
  user_row = np.array(df[df['User'] == user])
  bern_params = user_row[0][2:2 + param_dim]
  normal_mean_params = user_row[0][2 + param_dim:2 + 2 * param_dim]
  sigma_u = user_row[0][-1]

  # poisson parameters, bernouilli parameters
  return bern_params, normal_mean_params, sigma_u

def get_zero_infl_params_for_user(user, df, env_type='stat'):
  param_dim = 5 if env_type == 'stat' else 6
  user_row = np.array(df[df['User'] == user])
  bern_params = user_row[0][2:2 + param_dim]
  poisson_params = user_row[0][2 + param_dim:]

  # poisson parameters, bernouilli parameters
  return bern_params, poisson_params

"""### Functions For Environment Models
---
"""

def linear_model(state, params):
  return state @ params

def construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          bern_params, \
                                          mu_params, \
                                          sigma_u, \
                                          transform_type, \
                                          effect_func_bern=lambda user, state : 0, \
                                          effect_func_norm=lambda user, state : 0):
  bern_linear_comp = linear_model(state, bern_params)
  if (action == 1):
    bern_linear_comp += effect_func_bern(user, state)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
    # normal transform component
    x_mu = linear_model(state, mu_params)
    if (action == 1):
      x_mu += effect_func_norm(user, state)
    if transform_type == "sqrt":
      sample = norm.rvs(loc=x_mu, scale=sigma_u)
      sample = sample**2
    else:
      sample = np.random.lognormal(x_mu, sigma_u)

    # we round to the nearest integer to produce brushing duration in seconds
    return int(sample)

  else:
    return 0

def construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          bern_params, \
                                          poisson_params, \
                                          effect_func_bern=lambda user, state : 0, \
                                          effect_func_poisson=lambda user, state : 0):
  bern_linear_comp = linear_model(state, bern_params)
  if (action == 1):
    bern_linear_comp += effect_func_bern(user, state)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
    # poisson component
    x = linear_model(state, poisson_params)
    if (action == 1):
      x += effect_func_poisson(user, state)

    l = np.exp(x)
    sample = poisson.rvs(l)

    return sample

  else:
    return 0

stat_sqrt_norm_model = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[2]), \
                                          transform_type="sqrt")

stat_log_norm_model = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[2]), \
                                          transform_type="log")

stat_zero_infl_pois_model = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[0], \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[1])

non_stat_sqrt_norm_model = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="sqrt")

non_stat_log_norm_model = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="log")

non_stat_zero_infl_pois_model = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[0], \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[1])

# loading pickles of which model class was chosen for which user
STAT_USER_MODELS = pickle.load(open("../sim_env_data/stat_user_models.p", "rb"))
NON_STAT_USER_MODELS = pickle.load(open("../sim_env_data/non_stat_user_models.p", "rb"))

"""## Constructing the Effect Sizes
---
"""

bern_param_titles = ['Time.of.Day.Bern', \
                     'Prior.Day.Total.Brush.Time.norm.Bern', \
                     'Proportion.Brushed.In.Past.7.Days.Bern', \
                     'Day.Type.Bern']

norm_transform_param_titles = ['Time.of.Day.Mu', \
                               'Prior.Day.Total.Brush.Time.norm.Mu', \
                               'Proportion.Brushed.In.Past.7.Days.Mu', \
                               'Day.Type.Mu']

poisson_param_titles = ['Time.of.Day.Poisson', \
                        'Prior.Day.Total.Brush.Time.norm.Poisson', \
                     'Proportion.Brushed.In.Past.7.Days.Poisson', \
                     'Day.Type.Poisson']

# mean of each model class group component
hurdle_bern_mean = -1.0 * np.min(np.mean(np.abs(np.array([STAT_SQRT_NORM_DF[title] for title in bern_param_titles])), axis=1)) / 2
hurdle_sqrt_mean = np.min(np.mean(np.abs(np.array([STAT_SQRT_NORM_DF[title] for title in norm_transform_param_titles])), axis=1)) / 2
hurdle_log_mean = np.min(np.mean(np.abs(np.array([STAT_LOG_NORM_DF[title] for title in norm_transform_param_titles])), axis=1)) / 2

zero_infl_bern_mean = -1.0 * np.min(np.mean(np.abs(np.array([STAT_ZERO_INFL_POIS_DF[title] for title in bern_param_titles])), axis=1)) / 2
zero_infl_poisson_mean = np.min(np.mean(np.abs(np.array([STAT_ZERO_INFL_POIS_DF[title] for title in poisson_param_titles])), axis=1)) / 2

# large effect size means
large_hurdle_bern_mean = -1.0 * np.mean(np.mean(np.abs(np.array([STAT_SQRT_NORM_DF[title] for title in bern_param_titles])), axis=1))
large_hurdle_sqrt_mean = np.mean(np.mean(np.abs(np.array([STAT_SQRT_NORM_DF[title] for title in norm_transform_param_titles])), axis=1))
large_hurdle_log_mean = np.mean(np.mean(np.abs(np.array([STAT_LOG_NORM_DF[title] for title in norm_transform_param_titles])), axis=1))

large_zero_infl_bern_mean = -1.0 * np.mean(np.mean(np.abs(np.array([STAT_ZERO_INFL_POIS_DF[title] for title in bern_param_titles])), axis=1))
large_zero_infl_poisson_mean = np.mean(np.mean(np.abs(np.array([STAT_ZERO_INFL_POIS_DF[title] for title in poisson_param_titles])), axis=1))

# std of each model class group
hurdle_bern_std = np.std(np.mean(np.abs(np.array([STAT_SQRT_NORM_DF[title] for title in bern_param_titles])), axis=0))
hurdle_sqrt_std = np.std(np.mean(np.abs(np.array([STAT_SQRT_NORM_DF[title] for title in norm_transform_param_titles])), axis=0))
hurdle_log_std = np.std(np.mean(np.abs(np.array([STAT_LOG_NORM_DF[title] for title in norm_transform_param_titles])), axis=0))

zero_infl_bern_std = np.std(np.mean(np.abs(np.array([STAT_ZERO_INFL_POIS_DF[title] for title in bern_param_titles])), axis=0))
zero_infl_poisson_std = np.std(np.mean(np.abs(np.array([STAT_ZERO_INFL_POIS_DF[title] for title in poisson_param_titles])), axis=0))

## simulating the effect sizes per user ##

np.random.seed(1)
# sample from Normal
user_norm_bern_effect_sizes = np.random.normal(loc=hurdle_bern_mean, scale=hurdle_bern_std, size=len(USERS_SESSIONS.keys()))
user_norm_sqrt_effect_sizes = np.random.normal(loc=hurdle_sqrt_mean, scale=hurdle_sqrt_std, size=len(USERS_SESSIONS.keys()))
user_norm_log_effect_sizes = np.random.normal(loc=hurdle_log_mean, scale=hurdle_log_std, size=len(USERS_SESSIONS.keys()))

user_zero_infl_bern_effect_sizes = np.random.normal(loc=zero_infl_bern_mean, scale=zero_infl_bern_std, size=len(USERS_SESSIONS.keys()))
user_zero_infl_poisson_effect_sizes = np.random.normal(loc=zero_infl_poisson_mean, scale=zero_infl_poisson_std, size=len(USERS_SESSIONS.keys()))

# large effect sizes
user_norm_bern_large_effect_sizes = np.random.normal(loc=large_hurdle_bern_mean, scale=hurdle_bern_std, size=len(USERS_SESSIONS.keys()))
user_norm_sqrt_large_effect_sizes = np.random.normal(loc=large_hurdle_sqrt_mean, scale=hurdle_sqrt_std, size=len(USERS_SESSIONS.keys()))
user_norm_log_large_effect_sizes = np.random.normal(loc=large_hurdle_log_mean, scale=hurdle_log_std, size=len(USERS_SESSIONS.keys()))

user_zero_infl_bern_large_effect_sizes = np.random.normal(loc=large_zero_infl_bern_mean, scale=zero_infl_bern_std, size=len(USERS_SESSIONS.keys()))
user_zero_infl_poisson_large_effect_sizes = np.random.normal(loc=large_zero_infl_poisson_mean, scale=zero_infl_poisson_std, size=len(USERS_SESSIONS.keys()))

# building effect sizes dictionary
HURDLE_BERN_EFFECT_SIZES = {}
HURDLE_SQRT_EFFECT_SIZES = {}
HURDLE_LOG_EFFECT_SIZES = {}
ZERO_INFL_BERN_EFFECT_SIZE = {}
ZERO_INFL_POISSON_EFFECT_SIZES = {}

for i, user_id in enumerate(USERS_SESSIONS.keys()):
  HURDLE_BERN_EFFECT_SIZES[user_id] = user_norm_bern_effect_sizes[i]
  HURDLE_SQRT_EFFECT_SIZES[user_id] = user_norm_sqrt_effect_sizes[i]
  HURDLE_LOG_EFFECT_SIZES[user_id] = user_norm_log_effect_sizes[i]

  ZERO_INFL_BERN_EFFECT_SIZE[user_id] = user_zero_infl_bern_effect_sizes[i]
  ZERO_INFL_POISSON_EFFECT_SIZES[user_id] = user_zero_infl_poisson_effect_sizes[i]

# large effect sizes
LARGE_HURDLE_BERN_EFFECT_SIZES = {}
LARGE_HURDLE_SQRT_EFFECT_SIZES = {}
LARGE_HURDLE_LOG_EFFECT_SIZES = {}
LARGE_ZERO_INFL_BERN_EFFECT_SIZE = {}
LARGE_ZERO_INFL_POISSON_EFFECT_SIZES = {}

for i, user_id in enumerate(USERS_SESSIONS.keys()):
  LARGE_HURDLE_BERN_EFFECT_SIZES[user_id] = user_norm_bern_large_effect_sizes[i]
  LARGE_HURDLE_SQRT_EFFECT_SIZES[user_id] = user_norm_sqrt_large_effect_sizes[i]
  LARGE_HURDLE_LOG_EFFECT_SIZES[user_id] = user_norm_log_large_effect_sizes[i]

  LARGE_ZERO_INFL_BERN_EFFECT_SIZE[user_id] = user_zero_infl_bern_large_effect_sizes[i]
  LARGE_ZERO_INFL_POISSON_EFFECT_SIZES[user_id] = user_zero_infl_poisson_large_effect_sizes[i]

## USER-SPECIFIC EFFECT SIZES ##
# Context-Aware with all features same as baseline features excpet for Prop. Non-Zero Brushing In Past 7 Days
# which is of index 2 for stat models and of index 3 for non stat models
stat_user_spec_effect_func_hurdle_bern = lambda user, state: np.array(4 * [HURDLE_BERN_EFFECT_SIZES[user]]) @ np.delete(state, 2)
stat_user_spec_effect_func_hurdle_sqrt = lambda user, state: np.array(4 * [HURDLE_SQRT_EFFECT_SIZES[user]]) @ np.delete(state, 2)
stat_user_spec_effect_func_hurdle_log = lambda user, state: np.array(4 * [HURDLE_LOG_EFFECT_SIZES[user]]) @ np.delete(state, 2)
stat_user_spec_effect_func_zero_infl_bern = lambda user, state: np.array(4 * [ZERO_INFL_BERN_EFFECT_SIZE[user]]) @ np.delete(state, 2)
stat_user_spec_effect_func_zero_infl_poisson = lambda user, state: np.array(4 * [ZERO_INFL_POISSON_EFFECT_SIZES[user]]) @ np.delete(state, 2)

non_stat_user_spec_effect_func_hurdle_bern = lambda user, state: np.array(5 * [HURDLE_BERN_EFFECT_SIZES[user]]) @ np.delete(state, 3)
non_stat_user_spec_effect_func_hurdle_sqrt = lambda user, state: np.array(5 * [HURDLE_SQRT_EFFECT_SIZES[user]]) @ np.delete(state, 3)
non_stat_user_spec_effect_func_hurdle_log = lambda user, state: np.array(5 * [HURDLE_LOG_EFFECT_SIZES[user]]) @ np.delete(state, 3)
non_stat_user_spec_effect_func_zero_infl_bern = lambda user, state: np.array(5 * [ZERO_INFL_BERN_EFFECT_SIZE[user]]) @ np.delete(state, 3)
non_stat_user_spec_effect_func_zero_infl_poisson = lambda user, state: np.array(5 * [ZERO_INFL_POISSON_EFFECT_SIZES[user]]) @ np.delete(state, 3)

# large effect sizes
large_stat_user_spec_effect_func_hurdle_bern = lambda user, state: np.array(4 * [LARGE_HURDLE_BERN_EFFECT_SIZES[user]]) @ np.delete(state, 2)
large_stat_user_spec_effect_func_hurdle_sqrt = lambda user, state: np.array(4 * [LARGE_HURDLE_SQRT_EFFECT_SIZES[user]]) @ np.delete(state, 2)
large_stat_user_spec_effect_func_hurdle_log = lambda user, state: np.array(4 * [LARGE_HURDLE_LOG_EFFECT_SIZES[user]]) @ np.delete(state, 2)
large_stat_user_spec_effect_func_zero_infl_bern = lambda user, state: np.array(4 * [LARGE_ZERO_INFL_BERN_EFFECT_SIZE[user]]) @ np.delete(state, 2)
large_stat_user_spec_effect_func_zero_infl_poisson = lambda user, state: np.array(4 * [LARGE_ZERO_INFL_POISSON_EFFECT_SIZES[user]]) @ np.delete(state, 2)

large_non_stat_user_spec_effect_func_hurdle_bern = lambda user, state: np.array(5 * [LARGE_HURDLE_BERN_EFFECT_SIZES[user]]) @ np.delete(state, 3)
large_non_stat_user_spec_effect_func_hurdle_sqrt = lambda user, state: np.array(5 * [LARGE_HURDLE_SQRT_EFFECT_SIZES[user]]) @ np.delete(state, 3)
large_non_stat_user_spec_effect_func_hurdle_log = lambda user, state: np.array(5 * [LARGE_HURDLE_LOG_EFFECT_SIZES[user]]) @ np.delete(state, 3)
large_non_stat_user_spec_effect_func_zero_infl_bern = lambda user, state: np.array(5 * [LARGE_ZERO_INFL_BERN_EFFECT_SIZE[user]]) @ np.delete(state, 3)
large_non_stat_user_spec_effect_func_zero_infl_poisson = lambda user, state: np.array(5 * [LARGE_ZERO_INFL_POISSON_EFFECT_SIZES[user]]) @ np.delete(state, 3)

# Stationary
stat_sqrt_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: stat_user_spec_effect_func_hurdle_sqrt(user, state))

# large
large_stat_sqrt_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: large_stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: large_stat_user_spec_effect_func_hurdle_sqrt(user, state))

stat_log_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: stat_user_spec_effect_func_hurdle_log(user, state))

# large
large_stat_log_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: large_stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: large_stat_user_spec_effect_func_hurdle_log(user, state))

stat_zero_infl_pois_user_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[0], \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[1], \
                                          effect_func_bern=lambda user, state: stat_user_spec_effect_func_zero_infl_bern(user, state), \
                                          effect_func_poisson=lambda user, state: stat_user_spec_effect_func_zero_infl_poisson(user, state))

# large
large_stat_zero_infl_pois_user_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[0], \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[1], \
                                          effect_func_bern=lambda user, state: large_stat_user_spec_effect_func_zero_infl_bern(user, state), \
                                          effect_func_poisson=lambda user, state: large_stat_user_spec_effect_func_zero_infl_poisson(user, state))

# Non-Stationary
non_stat_sqrt_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: non_stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: non_stat_user_spec_effect_func_hurdle_sqrt(user, state))

# large
large_non_stat_sqrt_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: large_non_stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: large_non_stat_user_spec_effect_func_hurdle_sqrt(user, state))

non_stat_log_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: non_stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: non_stat_user_spec_effect_func_hurdle_log(user, state))

# large
large_non_stat_log_user_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: large_non_stat_user_spec_effect_func_hurdle_bern(user, state), \
                                          effect_func_norm=lambda user, state: large_non_stat_user_spec_effect_func_hurdle_log(user, state))

non_stat_zero_infl_pois_user_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[0], \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[1], \
                                          effect_func_bern=lambda user, state: non_stat_user_spec_effect_func_zero_infl_bern(user, state), \
                                          effect_func_poisson=lambda user, state: non_stat_user_spec_effect_func_zero_infl_poisson(user, state))

# large
large_non_stat_zero_infl_pois_user_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[0], \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[1], \
                                          effect_func_bern=lambda user, state: large_non_stat_user_spec_effect_func_zero_infl_bern(user, state), \
                                          effect_func_poisson=lambda user, state: large_non_stat_user_spec_effect_func_zero_infl_poisson(user, state))

## POPULATION EFFECT SIZES ##
# Stationary
stat_sqrt_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: np.array(4 * [hurdle_bern_mean]) @ np.delete(state, 2), \
                                          effect_func_norm=lambda user, state: np.array(4 * [hurdle_sqrt_mean]) @ np.delete(state, 2))

# large
large_stat_sqrt_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_SQRT_NORM_DF)[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: np.array(4 * [large_hurdle_bern_mean]) @ np.delete(state, 2), \
                                          effect_func_norm=lambda user, state: np.array(4 * [large_hurdle_sqrt_mean]) @ np.delete(state, 2))

stat_log_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: np.array(4 * [hurdle_bern_mean]) @ np.delete(state, 2), \
                                          effect_func_norm=lambda user, state: np.array(4 * [hurdle_log_mean]) @ np.delete(state, 2))

# large
large_stat_log_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[0], \
                                          get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[1], \
                                          abs(get_norm_transform_params_for_user(user, STAT_LOG_NORM_DF)[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: np.array(4 * [large_hurdle_bern_mean]) @ np.delete(state, 2), \
                                          effect_func_norm=lambda user, state: np.array(4 * [large_hurdle_log_mean]) @ np.delete(state, 2))

stat_zero_infl_pois_pop_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[0], \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[1], \
                                          effect_func_bern=lambda user, state: np.array(4 * [zero_infl_bern_mean]) @ np.delete(state, 2), \
                                          effect_func_poisson=lambda user, state: np.array(4 * [zero_infl_poisson_mean]) @ np.delete(state, 2))

# large
large_stat_zero_infl_pois_pop_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[0], \
                                          get_zero_infl_params_for_user(user, STAT_ZERO_INFL_POIS_DF)[1], \
                                          effect_func_bern=lambda user, state: np.array(4 * [large_zero_infl_bern_mean]) @ np.delete(state, 2), \
                                          effect_func_poisson=lambda user, state: np.array(4 * [large_zero_infl_poisson_mean]) @ np.delete(state, 2))

# Non-Stationary
non_stat_sqrt_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: np.array(5 * [hurdle_bern_mean]) @ np.delete(state, 3), \
                                          effect_func_norm=lambda user, state: np.array(5 * [hurdle_sqrt_mean]) @ np.delete(state, 3))
# large
large_non_stat_sqrt_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_SQRT_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="sqrt", \
                                          effect_func_bern=lambda user, state: np.array(5 * [large_hurdle_bern_mean]) @ np.delete(state, 3), \
                                          effect_func_norm=lambda user, state: np.array(5 * [large_hurdle_sqrt_mean]) @ np.delete(state, 3))

non_stat_log_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: np.array(5 * [hurdle_bern_mean]) @ np.delete(state, 3), \
                                          effect_func_norm=lambda user, state: np.array(5 * [hurdle_log_mean]) @ np.delete(state, 3))

# large
large_non_stat_log_pop_effect_size = lambda user, state, action: construct_norm_transform_sim_model_and_sample(user, state, action, \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[0], \
                                          get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[1], \
                                          abs(get_norm_transform_params_for_user(user, NON_STAT_LOG_NORM_DF, env_type='non_stat')[2]), \
                                          transform_type="log", \
                                          effect_func_bern=lambda user, state: np.array(5 * [large_hurdle_bern_mean]) @ np.delete(state, 3), \
                                          effect_func_norm=lambda user, state: np.array(5 * [large_hurdle_log_mean]) @ np.delete(state, 3))

non_stat_zero_infl_pois_pop_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[0], \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[1], \
                                          effect_func_bern=lambda user, state: np.array(5 * [zero_infl_bern_mean]) @ np.delete(state, 3), \
                                          effect_func_poisson=lambda user, state: np.array(5 * [zero_infl_poisson_mean]) @ np.delete(state, 3))

# large
large_non_stat_zero_infl_pois_pop_effect_size = lambda user, state, action: construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[0], \
                                          get_zero_infl_params_for_user(user, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')[1], \
                                          effect_func_bern=lambda user, state: np.array(5 * [large_zero_infl_bern_mean]) @ np.delete(state, 3), \
                                          effect_func_poisson=lambda user, state: np.array(5 * [large_zero_infl_poisson_mean]) @ np.delete(state, 3))

"""### Creating Simulation Environment Objects
---
"""

class SimulationEnvironment():
    def __init__(self, env_type, process_env_state, all_user_states, all_user_reward_models, users_list):
        # String: stat or non-stat
        self.env_type = env_type
        # Func
        self.process_env_state = process_env_state
        # Dict: key: String user_id, val: states
        self.all_user_states = all_user_states
        # Dict: key: String user_id, val: Fuction model_class reward generating function
        self.all_user_reward_models = all_user_reward_models
        # List: users in the environment (can repeat)
        self.users_list = users_list

    def generate_rewards(self, user_id, state, action):
        return self.all_user_reward_models[user_id](user_id, state, action)

    def get_states_for_user(self, user_id):
        return self.all_user_states[user_id]

    def get_env_type(self):
        return self.env_type

    def get_users(self):
        return self.users_list

# STATIONARY ENV. WITH HETEROGENEOUS EFFECT SIZE
stat_het_eff_reward_funcs = []
large_het_eff_reward_funcs = []
for val in STAT_USER_MODELS.values():
    if val == 'zero_infl':
        reward_func = stat_zero_infl_pois_user_effect_size
        large_het_eff_reward_funcs.append(large_stat_zero_infl_pois_user_effect_size)
    elif val == 'sqrt_norm':
        reward_func = stat_sqrt_user_effect_size
        large_het_eff_reward_funcs.append(large_stat_sqrt_user_effect_size)
    elif val == 'log_norm':
        reward_func = stat_log_user_effect_size
        large_het_eff_reward_funcs.append(large_stat_log_user_effect_size)
    stat_het_eff_reward_funcs.append(reward_func)

STAT_HET_EFF_REWARD_MODELS = dict(zip(STAT_USER_MODELS.keys(), stat_het_eff_reward_funcs))
LARGE_STAT_HET_EFF_REWARD_MODELS = dict(zip(STAT_USER_MODELS.keys(), large_het_eff_reward_funcs))

# NON-STATIONARY ENV. WITH HETEROGENEOUS EFFECT SIZE
non_stat_het_eff_reward_funcs = []
large_non_stat_het_eff_reward_funcs = []
for val in NON_STAT_USER_MODELS.values():
    if val == 'zero_infl':
        reward_func = non_stat_zero_infl_pois_user_effect_size
        large_non_stat_het_eff_reward_funcs.append(large_non_stat_zero_infl_pois_user_effect_size)
    elif val == 'sqrt_norm':
        reward_func = non_stat_sqrt_user_effect_size
        large_non_stat_het_eff_reward_funcs.append(large_non_stat_sqrt_user_effect_size)
    elif val == 'log_norm':
        reward_func = non_stat_log_user_effect_size
        large_non_stat_het_eff_reward_funcs.append(large_non_stat_log_user_effect_size)
    non_stat_het_eff_reward_funcs.append(reward_func)

NON_STAT_HET_EFF_REWARD_MODELS = dict(zip(NON_STAT_USER_MODELS.keys(), non_stat_het_eff_reward_funcs))
LARGE_NON_STAT_HET_EFF_REWARD_MODELS = dict(zip(NON_STAT_USER_MODELS.keys(), large_non_stat_het_eff_reward_funcs))

# STATIONARY ENV. WITH POPULATION EFFECT SIZE
stat_pop_eff_reward_funcs = []
large_stat_pop_eff_reward_funcs = []
for val in STAT_USER_MODELS.values():
    if val == 'zero_infl':
        reward_func = stat_zero_infl_pois_pop_effect_size
        large_stat_pop_eff_reward_funcs.append(large_stat_zero_infl_pois_pop_effect_size)
    elif val == 'sqrt_norm':
        reward_func = stat_sqrt_pop_effect_size
        large_stat_pop_eff_reward_funcs.append(large_stat_sqrt_pop_effect_size)
    elif val == 'log_norm':
        reward_func = stat_log_pop_effect_size
        large_stat_pop_eff_reward_funcs.append(large_stat_log_pop_effect_size)
    stat_pop_eff_reward_funcs.append(reward_func)

STAT_POP_EFF_REWARD_MODELS = dict(zip(STAT_USER_MODELS.keys(), stat_pop_eff_reward_funcs))
LARGE_STAT_POP_EFF_REWARD_MODELS = dict(zip(STAT_USER_MODELS.keys(), large_stat_pop_eff_reward_funcs))

# NON-STATIONARY ENV. WITH POPULATION EFFECT SIZE
non_stat_pop_eff_reward_funcs = []
large_non_stat_pop_eff_reward_funcs = []
for val in NON_STAT_USER_MODELS.values():
    if val == 'zero_infl':
        reward_func = non_stat_zero_infl_pois_pop_effect_size
        large_non_stat_pop_eff_reward_funcs.append(large_non_stat_zero_infl_pois_pop_effect_size)
    elif val == 'sqrt_norm':
        reward_func = non_stat_sqrt_pop_effect_size
        large_non_stat_pop_eff_reward_funcs.append(large_non_stat_sqrt_pop_effect_size)
    elif val == 'log_norm':
        reward_func = non_stat_log_pop_effect_size
        large_non_stat_pop_eff_reward_funcs.append(large_non_stat_log_pop_effect_size)
    non_stat_pop_eff_reward_funcs.append(reward_func)

NON_STAT_POP_EFF_REWARD_MODELS = dict(zip(NON_STAT_USER_MODELS.keys(), non_stat_pop_eff_reward_funcs))
LARGE_NON_STAT_POP_EFF_REWARD_MODELS = dict(zip(NON_STAT_USER_MODELS.keys(), large_non_stat_pop_eff_reward_funcs))

STAT_HET_EFF_ENV = lambda users_list: SimulationEnvironment('stat', stat_process_env_state, USERS_SESSIONS, STAT_HET_EFF_REWARD_MODELS, users_list)
NON_STAT_HET_EFF_ENV = lambda users_list: SimulationEnvironment('non_stat', non_stat_process_env_state, USERS_SESSIONS_NON_STATIONARITY, NON_STAT_HET_EFF_REWARD_MODELS, users_list)
STAT_POP_EFF_ENV = lambda users_list: SimulationEnvironment('stat', stat_process_env_state, USERS_SESSIONS, STAT_POP_EFF_REWARD_MODELS, users_list)
NON_STAT_POP_EFF_ENV = lambda users_list: SimulationEnvironment('non_stat', non_stat_process_env_state, USERS_SESSIONS_NON_STATIONARITY, NON_STAT_POP_EFF_REWARD_MODELS, users_list)

# large
LARGE_STAT_HET_EFF_ENV = lambda users_list: SimulationEnvironment('stat', stat_process_env_state, USERS_SESSIONS, LARGE_STAT_HET_EFF_REWARD_MODELS, users_list)
LARGE_NON_STAT_HET_EFF_ENV = lambda users_list: SimulationEnvironment('non_stat', non_stat_process_env_state, USERS_SESSIONS_NON_STATIONARITY, LARGE_NON_STAT_HET_EFF_REWARD_MODELS, users_list)
LARGE_STAT_POP_EFF_ENV = lambda users_list: SimulationEnvironment('stat', stat_process_env_state, USERS_SESSIONS, LARGE_STAT_POP_EFF_REWARD_MODELS, users_list)
LARGE_NON_STAT_POP_EFF_ENV = lambda users_list: SimulationEnvironment('non_stat', non_stat_process_env_state, USERS_SESSIONS_NON_STATIONARITY, LARGE_NON_STAT_POP_EFF_REWARD_MODELS, users_list)
