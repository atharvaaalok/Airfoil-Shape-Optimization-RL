import gymnasium as gym
from gymnasium import spaces

import numpy as np

from ..CFD.Aerodynamics import Aerodynamics


NEGATIVE_REWARD = -100.0


class CFD_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "no_display"], "render_fps": 4}

    def __init__(self, s0, idx_to_change, max_iterations, a_scaling, valid_states_file_path, use_delta_r):
        super(CFD_Env, self).__init__()
        self.action_space = spaces.Box(low = -1, high = 1, shape = (len(idx_to_change) * 2,), dtype = np.float32)
        self.observation_space = spaces.Box(low = -1, high = 2, shape = s0.flatten().shape, dtype = np.float32)

        self.s = s0
        self.idx_to_change = idx_to_change
        self.a_scaling = a_scaling
        self.valid_states_file_path = valid_states_file_path
        self.iter = 0
        self.max_iterations = max_iterations

        self.use_delta_r = use_delta_r
        self.prev_reward = 0
        self.new_reward = 0


    def step(self, action):
        action = action.reshape(-1, 2)
        s_new = self.s
        s_new[self.idx_to_change, :] = s_new[self.idx_to_change, :] + action * self.a_scaling
        self.s = s_new

        terminated = False
        truncated = False

        # Generate reward
        self.new_reward = self._generate_reward(s_new)

        if self.use_delta_r:
            reward = self.new_reward - self.prev_reward
            self.prev_reward = self.new_reward
        else:
            reward = self.new_reward
            self.prev_reward = self.new_reward
        
        
        observation = s_new.flatten()
        info = {}

        self.iter += 1
        if self.iter == self.max_iterations:
            terminated = True
        else:
            terminated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed = None, options = None):
        np.random.seed(seed)
        # Choose a random valid initial state from file
        file_path = self.valid_states_file_path
        s_new = read_random_line(file_path).reshape(-1, 2).astype(np.float32)

        self.s = s_new

        # Set state's reward
        self.prev_reward = self._generate_reward(s_new)
        self.new_reward = 0

        observation = s_new.flatten()
        # observation = s_new[self.idx_to_change, :].flatten()
        info = {}

        self.iter = 0

        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def _generate_reward(self, s):
        # Generate reward
        airfoil_name = 'air' + str(np.random.rand(1))[3:-1]
        airfoil_coordinates = s
        airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)
        Reynolds_num = 1e6
        L_by_D_ratio = airfoil.get_L_by_D(Reynolds_num)

        if L_by_D_ratio == None:
            L_by_D_ratio = NEGATIVE_REWARD
        
        return L_by_D_ratio




# Function to read a random line from the file and convert it into a NumPy vector
def read_random_line(file_path):
    with open(file_path, 'r') as file:
        # Count the total number of lines in the file
        num_lines = sum(1 for line in file)
    
    # Generate a random line number within the range of total lines
    random_line_number = np.random.randint(0, num_lines - 1)
    
    # Read the selected random line from the file
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            if line_num == random_line_number:
                # Convert the line into a NumPy vector
                numpy_vector = np.fromstring(line, dtype = float, sep=' ')
                return numpy_vector  # Return the NumPy vector