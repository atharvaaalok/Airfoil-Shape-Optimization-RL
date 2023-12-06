from stable_baselines3.common.env_checker import check_env
from .CFD_Gym_Env import CFD_Env

import numpy as np
import os

s0 = np.array([[1, 0], [0.75, 0.05], [0.625, 0.075], [0.5, 0.1], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.1], [0.625, -0.075], [0.75, -0.05], [1, 0]])
s0 = s0.astype(np.float32)
idx_to_change = [1, 2, 3, 4, 6, 7, 8, 9]
a_scaling = (1 / 1000)
valid_states_file_path = os.path.dirname(__file__) + '/Dataset/Arrays_as_rows.txt'


# Get the environment
env = CFD_Env(s0, idx_to_change, a_scaling, valid_states_file_path)
check_env(env)




# Also run the following

episodes = 10
for episode in range(episodes):
    terminated = False
    observation, info = env.reset()
    while not terminated:
        random_action = env.action_space.sample()
        print('action:', random_action)
        observation, reward, terminated, truncated, info = env.step(random_action)
        print('reward:', reward)
    print()
