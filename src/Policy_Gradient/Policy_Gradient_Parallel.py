import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from .Helper import *
from .Trajectory import Trajectory

import concurrent.futures



def get_trajectory_rewards(SAS_list):
    reward_list = []
    for s, a, s_new in SAS_list:
        reward_list.append(generate_reward(s, a, s_new))
    return reward_list



if __name__ == '__main__':

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
    T = 5
    # Batch size
    N = 5
    # Total policy improvements
    epochs = 20

    # Set policy parameters and the MDP functions required to generate trajectories
    policy_params = {'policy_net': policy_net, 'Sigma': Sigma}
    MDP_functions = {'generate_action': generate_action, 'generate_next_state': generate_next_state, 'generate_reward': generate_reward}
    

    # Training loop
    for epoch in range(epochs):

        J = torch.tensor(0.0)

        trajectory_list = []
        # Generate training batch
        for i_traj in range(N):
            s0 = torch.tensor([[1, 0], [0.75, 0.05], [0.5, 0.10], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.10], [1, 0]])

            trajectory = Trajectory(s0, T, policy_params, MDP_functions, parallelize = True)
            trajectory_list.append(trajectory)
        
        # Get s, a, s_new pairs to calculate corresponding rewards in parallel
        SAS_list = [trajectory.get_SAS_list() for trajectory in trajectory_list]
        
        # For all trajectories calculate rewards
        with concurrent.futures.ProcessPoolExecutor(max_workers = 60) as executor:
            reward_lists = executor.map(get_trajectory_rewards, SAS_list)
        
        # Update the trajectory rewards
        for trajectory, rewards in zip(trajectory_list, reward_lists):
            trajectory.set_rewards(rewards)

        for trajectory in trajectory_list:
            # Get total log probability and total reward for the trajectory
            total_log_prob = trajectory.get_total_log_prob()
            total_reward = trajectory.get_total_reward()
            
            # Update J
            J += -1 * total_log_prob * total_reward
        
        J = (1 / N) * J

        # Update the policy and Sigma
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Episode {epoch + 1}/{epochs} | Policy Loss: {J.item()}")

    quit()

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