import numpy as np
import torch
from ..ML_Modules.NeuralNetwork import NeuralNetwork, Train_NN
from ..CFD.Aerodynamics import Aerodynamics
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Create state and action lists
s_list = []
a_list = []

# Generate initial state
s0 = torch.tensor([[1, 0], [0.75, 0.05], [0.5, 0.10], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.10], [0.75, -0.05], [1, 0]])
s_list.append(s0)

# Total number of points to represent the shape
pts_on_curve = s0.shape[0]
action_idx = [1, 2, 3, 5, 6, 7]
action_dim = len(action_idx)

# Plot the initial airfoil
airfoil_coordinates = s0.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], marker = 'o')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.show()


# Define the experiment run count
total_exp = 10
print('Running Experiments\n' + 30 * '-')
for i_exp in range(total_exp):
    print(f'Experiment Count: {i_exp + 1}')
    # Get the state
    s = s_list[i_exp]

    # Generate random actions to take
    num_actions = 10
    # Calculate the L/D ratio for each new state resulting from the actions and determine the best action
    Rewards = torch.zeros(num_actions + 1)
    a_temp = []

    # Define action scaling properties
    step = 0.0075
    p = 0.1
    jump = step / ((i_exp + 1) ** p)

    for i_a in range(num_actions + 1):
        # Generate actions and make sure that actions are only taken for the movable points and the fixed points at (1, 0) and (0, 0) are untouched
        # Also generate an action that does not change the state (delta_x/y = 0) to ensure that if the current state is best, don't change it
        if i_a == num_actions:
            a = torch.zeros(pts_on_curve, 2)
            a_temp.append(a)
        else:        
            a = torch.zeros(pts_on_curve, 2)
            a[action_idx, :] = jump * torch.rand(action_dim, 2)
            a_temp.append(a)

        # Get new states from current state and the generated action
        s_prime = s + a

        # Get the airfoil coordinates
        airfoil_coordinates = s_prime.numpy()

        # Create airfoil object to analyze properties
        airfoil_name = f'my_airfoil{i_exp}{i_a}'
        airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)

        # Get L/D ratio
        Reynolds_num = 1e6
        reward = airfoil.get_L_by_D(Reynolds_num)
        # print(reward)
        if reward == None:
            # If xfoil doesn't converge give a large negative reward
            reward = -1000
        Rewards[i_a] = reward

    idx_max_reward = torch.argmax(Rewards)
    a_best = a_temp[idx_max_reward]
    max_reward = Rewards[idx_max_reward].item()
    print(f'Max reward: {max_reward}\n')

    # Get the new state corresponding to the best action
    s_new = s + a_best

    # Add the action and the new state to the state and action lists
    a_list.append(a_best)
    s_list.append(s_new)

print()

# Plot the new airfoil
airfoil_coordinates = s_new.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], marker = 'o')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# Visualize the final airfoil in xfoil
airfoil_name = 'final_airfoil'
airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)
airfoil.visualize()

S_input_list = []
A_input_list = []

for i in range(len(a_list)):
    S_input_list.append(torch.cat((s_list[i][:, 0], s_list[i][:, 1])))
    A_input_list.append(torch.cat((a_list[i][:, 0], a_list[i][:, 1])))

# Train neural network using the states as input and the actions as the labeled outputs
S = torch.stack(S_input_list)
A = torch.stack(A_input_list)


# Specify the size of the neural network and instantiate an object
input_size = pts_on_curve * 2
output_size = pts_on_curve * 2
layer_size_list = [20, 20]

NN_model = NeuralNetwork(input_size, output_size, layer_size_list)

# Define hyperparameters
learning_rate = 0.01
training_epochs = 1000

# Train the neural network
Train_NN(S, A, NN_model, learning_rate, training_epochs)