import numpy as np

from ..CFD.Aerodynamics import Aerodynamics

import time
import os

NEGATIVE_REWARD = -50


# Generate next state given the current state and action
def generate_next_state(s_current, a_current):
    s_new = s_current + a_current
    return s_new

# Generate reward corresponding to the state
def generate_reward(s, a, s_new, airfoil_name = 'my_airfoil'):
    airfoil_name = str(np.random.rand(1))[3:-1]

    airfoil_name = 'air' + airfoil_name
     # Get coordinates of airfoil
    airfoil_coordinates = s_new

    # Create airfoil object to analyze properties
    airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)

    # Get lift-to-drag ratio
    Reynolds_num = 1e6
    L_by_D_ratio = airfoil.get_L_by_D(Reynolds_num)

    # If Xfoil did not converge give large negative reward
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



# File paths
directory_path = os.path.dirname(__file__)
numpy_arr_file_path = directory_path + '/Dataset/Arrays_as_rows_a_scaling_100.txt'
rewards_file_path = directory_path + '/Dataset/Rewards_as_rows_a_scaling_100.txt'

# Trajectory parameters
T = 15
N = 10



if __name__ == '__main__':

    start_time = time.perf_counter()

    # Initialize a state
    # s0 = np.array([[1, 0], [0.75, 0.05], [0.625, 0.075], [0.5, 0.1], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.1], [0.625, -0.075], [0.75, -0.05], [1, 0]])
    idx_to_change = [1, 2, 3, 4, 6, 7, 8, 9]

    max_iterations = 5
    # s_new = s0

    for iteration in range(max_iterations):

        total_new_points = 0
        max_reward = 0
        state_list = []
        reward_list = []

        for i_N in range(N):
            # Get an initial state from file
            num_lines = sum(1 for _ in open(numpy_arr_file_path))
            s_new = read_random_line(numpy_arr_file_path).reshape(-1, 2)

            for t in range(T):
                # Get state
                s = s_new
                # Generate an action
                a = np.zeros(s.shape)
                a[idx_to_change, :] = np.random.rand(len(idx_to_change), 2) / 100

                # Get new state using this action
                s_new = generate_next_state(s, a)

                # Generate reward for the new state
                r = generate_reward(s, a, s_new)

                # If we achieved convergence record the state-reward pair
                if r > NEGATIVE_REWARD + 1:
                    # Add state-reward tuple to list
                    state_list.append(s_new.flatten())
                    reward_list.append(np.array([r]))
                    # Increment counter for total valid states generated
                    total_new_points += 1
                    if r > max_reward:
                        max_reward = r
        
        # Save data into a file and clear the arrays to save space
        # Save state-reward tuples to file
        with open(numpy_arr_file_path, 'a') as file1, open(rewards_file_path, 'a') as file2:
            np.savetxt(file1, state_list)
            np.savetxt(file2, reward_list)
        
        print(f'Total new points: {total_new_points}')
        print(f'Maximum reward: {max_reward}')
        print()


# Print finish time and the time taken per iteration
finish_time = time.perf_counter()
print(f'Total time taken for {max_iterations}: {finish_time - start_time}')
print(f'Time per iteration: {(finish_time - start_time) / (max_iterations * T * N)}')