import gymnasium as gym
from stable_baselines3 import PPO

from .CFD_Gym_Env import CFD_Env

import numpy as np
import os


algorithm_name = 'PPO'


models_dir = os.path.dirname(__file__) + '/Models/' + algorithm_name
log_dir = os.path.dirname(__file__) + '/Logs'


s0 = np.array([[1, 0], [0.75, 0.05], [0.625, 0.075], [0.5, 0.1], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.1], [0.625, -0.075], [0.75, -0.05], [1, 0]])
idx_to_change = [1, 2, 3, 4, 6, 7, 8, 9]
a_scaling = (1 / 1000)
valid_states_file_path = os.path.dirname(__file__) + '/Dataset/Arrays_as_rows.txt'


# Get the environment
training_env = CFD_Env(s0, idx_to_change, a_scaling, valid_states_file_path)
# Reset the environment
observation, info = training_env.reset()


# Train PPO on the environment
model = PPO('MlpPolicy', training_env, verbose = 1, tensorboard_log = log_dir)

TIMESTEPS = 5000
epochs = 40
for i in range(1, epochs):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name = algorithm_name, progress_bar = True)
    model.save(f'{models_dir}/{TIMESTEPS * i}')