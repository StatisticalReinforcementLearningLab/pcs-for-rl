# pcs-for-rl
Code for Designing Reinforcement Learning Algorithms for Digital Interventions: Pre-implementation Guidelines Paper

## Creating the Simulation Environment Variants
To create the environmnet variants we 1) fit a base model for each user and 2) impute effect sizes.

### Fitting Simulation Environment Base Models

We fit 3 environment base models (reward generating function under no intervention / action): using the ROBAS 2 data set found [here](https://github.com/ROBAS-UCLA/ROBAS.2/blob/master/inst/extdata/robas_2_data.csv).

The code for fitting the environment base models is in [fitting_environment_base_models.py](https://github.com/StatisticalReinforcementLearningLab/pcs-for-rl/blob/main/code/fitting_environment_base_models.py) and selecting the environment base model best fit for each ROBAS 2 user is in [selecting_environment_base_models.py](https://github.com/StatisticalReinforcementLearningLab/pcs-for-rl/blob/main/code/selecting_environment_base_models.py).

After fitting, the base model parameters and the selected base model for each ROBAS 2 user are saved to csv files in the `sim_env_data` folder.
Base model parameters follow the naming pattern `{environment_base_model}_params.csv`, and the selected base model for each user is a dictionary saved as a pickle file with the names `stat_user_models.p` and `non_stat_user_models.p`.

### Imputing Effect Sizes
The code for imputing effect sizes (reward under intervention / action) and defining simulation environment variants are found in [simulation_environments.py](https://github.com/StatisticalReinforcementLearningLab/pcs-for-rl/blob/main/code/simulation_environments.py)

## RL Algorithm Candidates
The RL algorithm candidates are defined in [rl_algorithm_candidates.py](https://github.com/StatisticalReinforcementLearningLab/pcs-for-rl/blob/main/code/rl_algorithm_candidates.py).

## Running Experiments
Experiment helper function are found in [rl_experiments.py](https://github.com/StatisticalReinforcementLearningLab/pcs-for-rl/blob/main/code/rl_experiments.py). Thre are two functions 1) `run_experiment` which runs the experiment without incremental recuirtment (assume all users have the same entry date) and 2) `run_incremental_recruitment_exp` which runs the experiment with incremental recruitment.

To run a single trial for an experiment, please call `python3 run_rl_experiment.py --sim_env_type=${SIM_ENV} --algorithm_candidate=${CANDIDATE} --seed=${SEED}`. (link for [run_rl_experiment.py](https://github.com/StatisticalReinforcementLearningLab/pcs-for-rl/blob/main/code/run_rl_experiment.py))  The `sim_env_type` denotes the simulation environment variant name, `algorithm_candidate` denotes the algorithm candidates and `seed` denotes the number that we will seed using `np.random.seed(seed_number)` before beginning the trial. 

Simulation Environmnet Variant Names:
`["stat_het_eff_env", "non_stat_het_eff_env", "stat_pop_eff_env", "non_stat_pop_eff_env", "large_stat_het_eff_env",  "large_stat_pop_eff_env", "large_non_stat_het_eff_env", "large_non_stat_pop_eff_env"]`

Note: for the paper, experiments were run using the "large" effect size environment variants.

Algorithm Names:
`["zip_k_1", "zip_k_4", "zip_k_full", "blr_k_1", "blr_k_4", "blr_k_full"]`

