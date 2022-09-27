# -*- coding: utf-8 -*-
"""
Selecting one model out of the three base model classes based on RMSE
"""

# pull packages
import numpy as np
import pandas as pd


from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import poisson
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

"""## Model Selection: Choosing the Best Model Based on RMSE
---

### Helper Functions
---
"""

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

get_zero_infl_params_for_user('P2_012', NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')

def bern_mean(w_b, X):
  q = sigmoid(X @ w_b)
  mean = 1 - q

  return mean

def poisson_mean(w_p, X):
  lam = np.exp(X @ w_p)

  return lam

def sqrt_norm_mean(w_mu, sigma_u, X):
  norm_mu = X @ w_mu
  mean = sigma_u**2 + norm_mu**2

  return mean

def log_norm_mean(w_mu, sigma_u, X):
  norm_mu = X @ w_mu
  mean = np.exp(norm_mu + (sigma_u**2 / 2))

  return mean

def compute_rmse_sqrt_norm(Xs, Ys, w_b, w_mu, sigma_u):
  mean_func = lambda x: bern_mean(w_b, x) * sqrt_norm_mean(w_mu, sigma_u, x)
  result = np.array([(Ys - mean_func(X))**2 for X in Xs])

  return np.sqrt(np.mean(result))

def compute_rmse_log_norm(Xs, Ys, w_b, w_mu, sigma_u):
  mean_func = lambda x: bern_mean(w_b, x) * log_norm_mean(w_mu, sigma_u, x)
  result = np.array([(Ys - mean_func(X))**2 for X in Xs])

  return np.sqrt(np.mean(result))

def compute_rmse_zero_infl(Xs, Ys, w_b, w_p):
  mean_func = lambda x: bern_mean(w_b, x) * poisson_mean(w_p, x)
  result = np.array([(Ys - mean_func(X))**2 for X in Xs])
  return np.sqrt(np.mean(result))

"""### Evaluations
---
"""

# sqrt, lognorm, zip
sqrt_norm = 'sqrt_norm'
log_norm = 'log_norm'
zero_infl = 'zero_infl'

MODEL_CLASS = {0: sqrt_norm, 1: log_norm, 2: zero_infl}

def choose_best_model(users_dict, users_rewards, sqrt_norm_df, log_norm_df, zero_infl_pois_df, env_type='stat'):
  USER_MODELS = {}
  USER_RMSES = {}

  for i, user in enumerate(users_dict.keys()):
      print("FOR USER: ", user)
      user_rmses = np.zeros(3)
      Xs = users_dict[user]
      Ys = users_rewards[user]
      ### HURDLE MODELS ###
      # Note: Sqrt Transform and Log Transform have the same Bern params
      hurdle_w_b = get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)[0]

      ## SQRT TRANSFORM ##
      sqrt_w_mu = get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)[1]
      sqrt_sigma_u = abs(get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)[2])
      user_rmses[0] = compute_rmse_sqrt_norm(Xs, Ys, hurdle_w_b, sqrt_w_mu, sqrt_sigma_u)
      ## LOG TRANSOFRM ##
      log_w_mu = get_norm_transform_params_for_user(user, log_norm_df, env_type)[1]
      log_sigma_u = abs(get_norm_transform_params_for_user(user, log_norm_df, env_type)[2])
      user_rmses[1] = compute_rmse_log_norm(Xs, Ys, hurdle_w_b, log_w_mu, log_sigma_u)
      ## 0-INFLATED MODELS ##
      zero_infl_w_b = get_zero_infl_params_for_user(user, zero_infl_pois_df, env_type)[0]
      w_p = get_zero_infl_params_for_user(user, zero_infl_pois_df, env_type)[1]
      user_rmses[2] = compute_rmse_zero_infl(Xs, Ys, zero_infl_w_b, w_p)
      print(MODEL_CLASS[np.argmin(user_rmses)])
      print("RMSEs: ", user_rmses)

      USER_MODELS[user] = MODEL_CLASS[np.argmin(user_rmses)]
      USER_RMSES[user] = user_rmses

  return USER_MODELS

STAT_USER_MODELS = choose_best_model(users_sessions, users_rewards, \
                                     STAT_SQRT_NORM_DF, STAT_LOG_NORM_DF, \
                                     STAT_ZERO_INFL_POIS_DF)

NON_STAT_USER_MODELS = choose_best_model(users_sessions_non_stationarity, users_rewards, \
                                     NON_STAT_SQRT_NORM_DF, NON_STAT_LOG_NORM_DF, \
                                     NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')

print("For Stationary Environment: ")
print("Num. Sqrt Model: ", len([x for x in STAT_USER_MODELS.values() if x == sqrt_norm]))
print("Num. Log Model: ", len([x for x in STAT_USER_MODELS.values() if x == log_norm]))
print("Num. 0-Inflated Poisson Model: ", len([x for x in STAT_USER_MODELS.values() if x == zero_infl]))

print("For Non-Stationary Environment: ")
print("Num. Sqrt Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == sqrt_norm]))
print("Num. Log Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == log_norm]))
print("Num. 0-Inflated Poisson Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == zero_infl]))

# pickling dicts
pickle.dump(STAT_USER_MODELS, open("../sim_env_data/stat_user_models.p", "wb"))
pickle.dump(NON_STAT_USER_MODELS, open("../sim_env_data/non_stat_user_models.p", "wb"))
