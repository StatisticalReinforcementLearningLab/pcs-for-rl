from rl_experiments import *
from absl import app
from absl import flags

# abseil ref: https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS
flags.DEFINE_string('sim_env_type', None, 'input the simulation environment type')
flags.DEFINE_string('algorithm_candidate', None, 'input the RL algorithm candidate type')
flags.DEFINE_integer('seed', None, 'seed for np.random.seed() to reproduce the trial')

NUM_TRIAL_USERS = 72
def get_user_list(study_idxs):
    user_list = [USER_INDICES[idx] for idx in study_idxs]

    return user_list

# parses argv to access FLAGS
def main(_argv):
    # draw different users per trial
    np.random.seed(FLAGS.seed)
    print("SEED: ", FLAGS.seed)
    # denotes a weekly update schedule
    UPDATE_CADENCE = 13
    STUDY_IDXS = np.random.choice(32, size=NUM_TRIAL_USERS)
    print(STUDY_IDXS)

    ## HANDLING RL ALGORITHM CANDIDATE ##
    candidate_type = FLAGS.algorithm_candidate
    print("PROCESSED CANDIDATE {}".format(candidate_type))
    if candidate_type == 'zip_k_1':
        cluster_size = 1
        alg_candidate = BayesianZeroInflatedPoisson(cluster_size, UPDATE_CADENCE)
    elif candidate_type == 'zip_k_4':
        cluster_size = 4
        alg_candidate = BayesianZeroInflatedPoisson(cluster_size, UPDATE_CADENCE)
    elif candidate_type == 'zip_k_full':
        cluster_size = NUM_TRIAL_USERS
        alg_candidate = BayesianZeroInflatedPoisson(cluster_size, UPDATE_CADENCE)
    elif candidate_type == 'blr_k_1':
        cluster_size = 1
        alg_candidate = BayesianLinearRegression(cluster_size, UPDATE_CADENCE)
    elif candidate_type == 'blr_k_4':
        cluster_size = 4
        alg_candidate = BayesianLinearRegression(cluster_size, UPDATE_CADENCE)
    elif candidate_type == 'blr_k_full':
        cluster_size = NUM_TRIAL_USERS
        alg_candidate = BayesianLinearRegression(cluster_size, UPDATE_CADENCE)
    else:
        print("ERROR: NO VALID CANDIDATE FOUND - ", candidate_type)

    # depending on the cluster size, we want to grab the cluster_num's cluster of users
    USERS_LIST = get_user_list(STUDY_IDXS)

    print(USERS_LIST)

    ## HANDLING SIMULATION ENVIRONMENT ##
    env_type = FLAGS.sim_env_type
    if env_type == 'stat_het_eff_env':
        environment_module = STAT_HET_EFF_ENV(USERS_LIST)
    elif env_type == 'non_stat_het_eff_env':
        environment_module = NON_STAT_HET_EFF_ENV(USERS_LIST)
    elif env_type == 'stat_pop_eff_env':
        environment_module = STAT_POP_EFF_ENV(USERS_LIST)
    elif env_type == 'non_stat_pop_eff_env':
        environment_module = NON_STAT_POP_EFF_ENV(USERS_LIST)
    elif env_type == 'large_stat_het_eff_env':
        environment_module = LARGE_STAT_HET_EFF_ENV(USERS_LIST)
    elif env_type == 'large_non_stat_het_eff_env':
        environment_module = LARGE_NON_STAT_HET_EFF_ENV(USERS_LIST)
    elif env_type == 'large_stat_pop_eff_env':
        environment_module = LARGE_STAT_POP_EFF_ENV(USERS_LIST)
    elif env_type == 'large_non_stat_pop_eff_env':
        environment_module = LARGE_NON_STAT_POP_EFF_ENV(USERS_LIST)
    else:
        print("ERROR: NO ENV_TYPE FOUND - ", env_type)

    print("PROCESSED ENV_TYPE {}".format(env_type))

    ## RUN EXPERIMENT ##
    # Full Pooling with Incremental Recruitment
    if (cluster_size == NUM_TRIAL_USERS):
        results = run_incremental_recruitment_exp(pre_process_users(USERS_LIST), alg_candidate, environment_module)
    else:
        results = run_experiment(alg_candidate, environment_module)
    pickling_location = '/n/home02/atrella/processed_pickle_results/{}_{}_{}_processed_result.p'.format(FLAGS.algorithm_candidate, FLAGS.sim_env_type, FLAGS.seed)

    ## results is a list of tuples where the first element of the tuple is user_id and the second element is a dictionary of values
    print("TRIAL DONE, PICKLING NOW")
    print("PICKLING TO: {}".format(pickling_location))
    with open(pickling_location, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    app.run(main)
