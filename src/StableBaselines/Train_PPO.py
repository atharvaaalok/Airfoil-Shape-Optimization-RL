import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .CFD_Gym_Env import CFD_Env

import numpy as np
import os


# Function that returns gym environments to use with vectorized environments
def make_env(s0, idx_to_change, max_iterations, a_scaling, valid_states_file_path, use_delta_r) -> gym.Env:

    # Get the environment
    env = CFD_Env(s0, idx_to_change, max_iterations, a_scaling, valid_states_file_path, use_delta_r)
    # Reset the environment
    observation, info = env.reset()

    return env



if __name__ == '__main__':

    # Parameters to mess with
    algorithm_name = 'PPO'

    s0 = np.array([[1, 0], [0.75, 0.05], [0.625, 0.075], [0.5, 0.1], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.1], [0.625, -0.075], [0.75, -0.05], [1, 0]])
    idx_to_change = [1, 2, 3, 4, 6, 7, 8, 9]
    a_scaling = (1 / 1000)
    valid_states_file_path = os.path.dirname(__file__) + '/Dataset/Arrays_as_rows.txt'
    MAX_ITERATIONS = 50

    parallelize = True
    num_cpu = 5

    use_custom_policy = True
    network_arch = [128, 64, 64]
    policy_kwargs = dict(net_arch = dict(pi = network_arch, vf = network_arch))

    use_delta_r = False

    CHECKPOINT_TIMESTEPS = 5000
    EPOCHS = 50


    if use_delta_r:
        algorithm_name = algorithm_name + '_DeltaR'

    if parallelize:
        algorithm_name = algorithm_name + '_Parallel'
        training_env = make_vec_env(lambda: make_env(s0, idx_to_change, MAX_ITERATIONS, a_scaling, valid_states_file_path, use_delta_r), n_envs = num_cpu, vec_env_cls = SubprocVecEnv)
    else:
        training_env = CFD_Env(s0, idx_to_change, MAX_ITERATIONS, a_scaling, valid_states_file_path, use_delta_r)
        # Reset the environment
        observation, info = training_env.reset()
    
    if use_custom_policy:
        algorithm_name = algorithm_name + '_PolicyArch' + '_'.join(map(str, network_arch))
    else:
        policy_kwargs = None

    

    # Get folders to save trained models and logs into
    models_dir = os.path.dirname(__file__) + '/Models/' + algorithm_name
    log_dir = os.path.dirname(__file__) + '/Logs'


    # Get the model
    model = PPO('MlpPolicy', training_env, verbose = 1, tensorboard_log = log_dir, policy_kwargs = policy_kwargs)

    # Train the model
    for i in range(1, EPOCHS):
        model.learn(total_timesteps = CHECKPOINT_TIMESTEPS, reset_num_timesteps = False, tb_log_name = algorithm_name, progress_bar = True)
        model.save(f'{models_dir}/{CHECKPOINT_TIMESTEPS * i}')