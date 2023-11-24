import numpy as np
import torch
from ML_Modules.NeuralNetwork import NeuralNetwork, Train_NN
from CFD_Evaluation.Aerodynamics_Resources import Aerodynamics
import matplotlib.pyplot as plt


# Create state and action lists
s_list = []
a_list = []

# Total number of points to represent the shape
pts_on_curve = 8

# Generate initial state
theta = torch.linspace(0, 2 * torch.pi, pts_on_curve)
r = 1
x0 = r * torch.cos(theta)
y0 = r * torch.sin(theta)

s0 = torch.column_stack((x0, y0))
s0[-1, :] = s0[0, :]
s_list.append(s0)

# Plot the initial airfoil
airfoil_coordinates = s0.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.show()

# Define the experiment run count
total_exp = 1
print('Running Experiments\n' + 30 * '-')
for i_exp in range(total_exp):
    print(f'Experiment Count: {i_exp + 1}')
    # Get the state
    s = s_list[i_exp]

    # Generate random actions to take
    num_actions = 5 
    # Calculate the L/D ratio for each new state resulting from the actions and determine the best action
    Rewards = torch.zeros(num_actions)
    a_temp = []

    # Define action scaling
    a_scaling = 0.1

    for i_a in range(num_actions):
        a = torch.rand(pts_on_curve, 2) * a_scaling
        # Actions for the first and the last point have to be the same to form a closed curve
        a[-1, :] = a[0, :]
        a_temp.append(a)

        # Get new states from current state and the generated action
        s_prime = s + a

        # Get the aerodynamic coordinates
        airfoil_coordinates = s_prime.numpy()

        # Create airfoil object to analyze properties
        airfoil_name = f'my_airfoil{i_exp}{i_a}'
        airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)
        print(airfoil)

        # Get L/D ratio
        Reynolds_num = 1e6
        reward = airfoil.get_L_by_D(Reynolds_num)
        print(reward)
        if reward == None:
            reward = 0
        Rewards[i_a] = reward

    a_best = a_temp[torch.argmax(Rewards)]

    # Get the new state corresponding to the best action
    s_new = s + a_best

    # Add the action and the new state to the state and action lists
    a_list.append(a_best)
    s_list.append(s_new)

print()

# Plot the new airfoil
airfoil_coordinates = s_new.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()


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