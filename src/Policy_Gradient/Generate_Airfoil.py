import torch
import numpy as np
import matplotlib.pyplot as plt

import os

from .Policy_Gradient import *
from .Helper import *
from .Trajectory import Trajectory


# Initialize the policy network with flexible hidden layers
policy_net = PolicyNetwork(state_dim, action_dim, layer_size_list)

# Define the covariance matrix Sigma as a trainable parameter
Sigma = nn.Parameter(torch.randn(action_dim, action_dim), requires_grad = True)

# Load progress if checkpoint is available
if os.path.exists(trained_model_path):
    policy_net, Sigma = load_checkpoint(trained_model_path, policy_net, Sigma)
else:
    print("Trained model doesn't exist. Train a model first by running Policy_Gradient.py")
          

# Set policy parameters and the MDP functions required to generate trajectories
policy_params = {'policy_net': policy_net, 'Sigma': Sigma}
MDP_functions = {'generate_action': generate_action, 'generate_next_state': generate_next_state, 'generate_reward': generate_reward}




# Now take actions according to the trained policy to generate an airfoil
### Change only the Total_improvements variable below in this file
Total_improvements = 30
# Run for long time to generate optimized airfoil - don't calculate rewards if the goal is just shape optimization
airfoil_gen_trajectory = Trajectory(s0, a_params, Total_improvements, policy_params, MDP_functions, calculate_rewards = False)
s_final = airfoil_gen_trajectory.SARS[-1].s_new


# Plot initial airfoil
airfoil_coordinates = s0.numpy()
plt.plot(airfoil_coordinates[:, 0], airfoil_coordinates[:, 1], marker = 'o')

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