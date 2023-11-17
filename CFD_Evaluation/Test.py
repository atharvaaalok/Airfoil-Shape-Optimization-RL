import numpy as np
import torch
import Aerodynamics
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
total_exp = 5
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
    a_scaling = 1

    for i_a in range(num_actions):
        print(i_a)
        a = torch.rand(pts_on_curve, 2)
        # Actions for the first and the last point have to be the same to form a closed curve
        a[-1, :] = a[0, :]
        a_temp.append(a)

        # Get new states from current state and the generated action
        s_prime = s + a

        # Get the aerodynamic coordinates
        airfoil_coordinates = s_prime.numpy()
        print(airfoil_coordinates)

        print('1')

        # Create airfoil object to analyze properties
        airfoil_name = 'my_airfoil'
        airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)

        print('2')

        # Get L/D ratio
        Reynolds_num = 1e6
        reward = airfoil.get_L_by_D(Reynolds_num)
        print('2.5')
        if reward == None:
            reward = 0
        Rewards[i_a] = reward

        print('3')
        print()

    a_best = a_temp[torch.argmax(Rewards)]

    # Get the new state corresponding to the best action
    s_new = s + a_best

    # Add the action and the new state to the state and action lists
    a_list.append(a_best)
    s_list.append(s_new)

print()