import os

import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt

import random

from .Helper import *
from .Trajectory import Trajectory, Generate_trajectories, add_valid_initial_states


# Define where to store training progress and the final trained model
checkpoint_path = os.path.dirname(__file__) + '/Progress_Checkpoint/Checkpoint.pth'
trained_model_path = os.path.dirname(__file__) + '/Progress_Checkpoint/Trained_Model.pth'
training_performance_plot_path = os.path.dirname(__file__) + '/Progress_Checkpoint/Total_Reward_vs_Epochs.png'

# Set seed for reproducibity
torch.manual_seed(42)
random.seed(42)



# Define an initial state to start training and then final airfoil shape optimization from
s0 = torch.tensor([[1, 0], [0.75, 0.05], [0.5, 0.10], [0.25, 0.05], [0, 0], [0.25, -0.05], [0.5, -0.10], [1, 0]])
a_params = {'idx_tochange': [1, 2, 3, 5, 6], 'a_scaling': (1 / 1000)}

# Define constants and hyperparameters
state_dim = 5 * 2
action_dim = 5 * 2
layer_size_list = [100, 100]

learning_rate_policy_net = 0.001
learning_rate_Sigma = 0.001

T = 3 # Episode length
N = 3 # Batch size - number of trajectories each of length T - Set equal to number of parallel workers
epochs = 20 # Total policy improvements - total training updates

# Set parallel compute to true if you want to generate trajectories in parallel
parallelize = False

# Set reward function
use_delta_LbyD = False

# Define whether to use causality and baseline
use_causality = False
use_baseline = True



if __name__ == '__main__':

    # Initialize the policy network with flexible hidden layers
    policy_net = PolicyNetwork(state_dim, action_dim, layer_size_list)

    # Define the covariance matrix Sigma as a trainable parameter
    Sigma = nn.Parameter(torch.randn(action_dim, action_dim), requires_grad = True)

    # Define the optimizer
    optimizer = optim.Adam([
        {'params': policy_net.parameters(), 'lr': learning_rate_policy_net},
        {'params': Sigma, 'lr': learning_rate_Sigma}
    ])
    

    # Prepare for Training
    # Initialize epoch to 0
    epoch = 0
    # Keep track of valid initial states to start trajectory generation from
    Valid_initial_states = [s0]
    # Make reward and epoch lists to plot the training process
    Total_Reward_list = []
    Epoch_list = []
    
    # Load progress if checkpoint is available
    if os.path.exists(checkpoint_path):
        epoch, policy_net, Sigma, optimizer, Valid_initial_states, Epoch_list, Total_Reward_list = load_checkpoint(checkpoint_path, policy_net, Sigma, optimizer, learning_rate_policy_net, learning_rate_Sigma, Valid_initial_states, Epoch_list, Total_Reward_list)

    # Set policy parameters and the MDP functions required to generate trajectories
    policy_params = {'policy_net': policy_net, 'Sigma': Sigma}
    MDP_functions = {'generate_action': generate_action, 'generate_next_state': generate_next_state, 'generate_reward': generate_reward}

    
    # Prepare plot for dynamic updating
    plt.ion() # Turn on interactive mode
    plt.xlabel('Epochs')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Epochs')
    plt.grid(True)
    plt.show()

    # print(f'Total valid initial states: {len(Valid_initial_states)}')
    while epoch < epochs:

        # Select initial states to start trajectory generation from. One s0 for each trajectory
        s0_list = random.choices(Valid_initial_states, k = N)

        # Generate trajectories - policy rollout
        trajectory_list = Generate_trajectories(s0_list, a_params, T, N, policy_params, MDP_functions, parallelize)

        # Update list of valid initial states
        add_valid_initial_states(trajectory_list, Valid_initial_states)

        # Define if to set reward to delta L/D instead of L/D
        for trajectory in trajectory_list:
            trajectory.use_delta_r(use = use_delta_LbyD)
        
        # Get total reward for all the trajectories combined
        Total_Reward = calculate_total_reward(trajectory_list)

        # Compute the gradient loss function and define whether to use causality and baseline
        J = calculate_gradient_objective(trajectory_list, causality = use_causality, baseline = use_baseline)

        # Update the policy network and Sigma
        optimizer.zero_grad()
        J.backward()
        optimizer.step()

        # Print progress and save models after every 5% progress
        if (epoch + 1) % (epochs // 20) == 0:
            # print(f"Episode {epoch + 1}/{epochs} | Policy Loss: {J.item()}")
            print(f"Episode {epoch + 1}/{epochs} | Total Reward: {Total_Reward.item()}")
            # print(f'Total valid initial states: {len(Valid_initial_states)}')
            Total_Reward_list.append(Total_Reward)
            Epoch_list.append(epoch + 1)
            save_checkpoint(checkpoint_path, epoch, policy_net, Sigma, optimizer, Valid_initial_states, Epoch_list, Total_Reward_list)
            plt.plot(Epoch_list, Total_Reward_list, '-o', color = 'b')
            plt.pause(0.1)
        
        # Update epoch
        epoch += 1

    # Upon finishing of training saved the trained model and delete the checkpoint file, also remove any pre-existing trained models
    if os.path.exists(trained_model_path):
        os.remove(trained_model_path)
    os.rename(checkpoint_path, trained_model_path)


    # Turn off plot interactive mode and save the plot
    plt.savefig(training_performance_plot_path)
    plt.ioff()