import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from .Helper import *
from .Trajectory import Trajectory



# Set seed for reproducibility
torch.manual_seed(42)

# Define constants and hyperparameters
state_dim = 5 * 2
action_dim = 5 * 2
layer_size_list = [50, 50]

# Initialize the policy network with flexible hidden layers
policy_net = PolicyNetwork(state_dim, action_dim, layer_size_list)

# Define the covariance matrix Sigma as a trainable parameter
Sigma = nn.Parameter(torch.randn(action_dim, action_dim), requires_grad = True)

# Optimizer
learning_rate = 0.001
optimizer = optim.Adam(list(policy_net.parameters()) + [Sigma], lr = learning_rate)


# Episode length
T = 10
# Batch size
N = 10
# Total policy improvements
epochs = 100

# Set policy parameters and the MDP functions required to generate trajectories
policy_params = {'policy_net': policy_net, 'Sigma': Sigma}
MDP_functions = {'generate_action': generate_action, 'generate_next_state': generate_next_state, 'generate_reward': generate_reward}


# Training loop
for epoch in range(epochs):

    J = torch.tensor(0.0)

    trajectory_list = []
    # Generate training batch
    for i_traj in range(N):
        rewards = []
        s0 = torch.tensor([[1, 0], [0.75, 0.05], [0.5, 0.10], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.10], [1, 0]])

        trajectory = Trajectory(s0, T, policy_params, MDP_functions)
        trajectory_list.append(trajectory)

    Total_Reward = calculate_total_reward(trajectory_list)
    J = calculate_gradient_objective(trajectory_list, causality = False, baseline = True)

    # Update the policy and Sigma
    optimizer.zero_grad()
    J.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        # print(f"Episode {epoch + 1}/{epochs} | Policy Loss: {J.item()}")
        print(f"Episode {epoch + 1}/{epochs} | Total Reward: {Total_Reward.item()}")


# Now take actions according to policy to generate an airfoil
# Initialize a state
s0 = torch.tensor([[1, 0], [0.75, 0.05], [0.5, 0.10], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.10], [1, 0]])

N_airfoil = 100
# Run for long time to generate optimized airfoil
final_trajectory = Trajectory(s0, N_airfoil, policy_params, MDP_functions)
s_final = final_trajectory.SARS[-1].s_new


# Plot initial airfoil
airfoil_coordinates = s0.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], marker = 'o')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# Plot final airfoil
airfoil_coordinates = s_final.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], marker = 'o')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

# Visualize the final airfoil in xfoil
airfoil_name = 'final_airfoil'
airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)
airfoil.visualize()