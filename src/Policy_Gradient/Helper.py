import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from ..CFD.Aerodynamics import Aerodynamics


# Generate next state given the current state and action
def generate_next_state(s_current, a_current):
    s_new = s_current + a_current
    return s_new

# Generate reward corresponding to the state
def generate_reward(s, a, s_new, airfoil_name = 'my_airfoil'):
    airfoil_name = str(np.random.rand(1))[3:9]

    airfoil_name = 'air' + airfoil_name
     # Get coordinates of airfoil
    airfoil_coordinates = s_new.cpu().numpy()

    # Create airfoil object to analyze properties
    airfoil = Aerodynamics.Airfoil(airfoil_coordinates, airfoil_name)

    # Get lift-to-drag ratio
    Reynolds_num = 1e6
    L_by_D_ratio = airfoil.get_L_by_D(Reynolds_num)

    # If Xfoil did not converge give large negative reward
    if L_by_D_ratio == None:
        L_by_D_ratio = -50
        
    return L_by_D_ratio


# Generate action given the state and the policy
def generate_action(s, policy_params):
    # Get the parameters
    policy_net = policy_params['policy_net']
    Sigma = policy_params['Sigma']

    idx_tochange = [1, 2, 3, 5, 6]
    s_nn = s[idx_tochange, :]
    s_nn = torch.cat((s_nn[:, 0], s_nn[:, 1]))

    # Get the mean action from the policy network
    mu = policy_net(s_nn)

    # Sample action from a normal distribution with mean mu and covariance Sigma
    cov_matrix = torch.mm(Sigma, Sigma.t()) # Ensure covariance matrix is positive semi-definite
    distribution = torch.distributions.MultivariateNormal(mu, covariance_matrix = cov_matrix)
    
    # Sample an action from the distribution
    a_nn_orig = distribution.sample()

    # Calculate the log probability of taking this action
    log_prob = distribution.log_prob(a_nn_orig)

    a_scaling = (1 / 1000)
    a_nn = a_scaling * a_nn_orig
    action_dim = a_nn.shape[0]
    a_nn = torch.stack((a_nn[:action_dim // 2], a_nn[action_dim // 2:]), dim = 1)

    a = torch.zeros_like(s)
    a[idx_tochange, :] = a_nn

    return (a, log_prob)



# Define the neural network for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size_list):
        super(PolicyNetwork, self).__init__()
        layer_sizes = [input_size] + layer_size_list + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def calculate_gradient_objective(trajectory_list, causality = False, baseline = False):

    log_prob_mat = torch.stack([trajectory.action_log_prob for trajectory in trajectory_list])
    reward_mat = torch.stack([trajectory.rewards for trajectory in trajectory_list])

    N = len(trajectory_list)

    if causality == False and baseline == False:
        total_log_prob = log_prob_mat.sum(dim = 1)
        total_reward = reward_mat.sum(dim = 1)
        J = -1 * (1 / N) * (total_log_prob * total_reward).sum()
    elif causality == True and baseline == False:
        cumulative_reward = reward_mat.cumsum(dim = 1)
        J = -1 * (1 / N) * (log_prob_mat * cumulative_reward).sum()
    elif causality == False and baseline == True:
        total_log_prob = log_prob_mat.sum(dim = 1)
        total_reward = reward_mat.sum(dim = 1)
        average_reward = total_reward.mean()
        J = -1 * (1 / N) * (total_log_prob * (total_reward - average_reward)).sum()
    elif causality == True and baseline == True:
        cumulative_reward = reward_mat.cumsum(dim = 1)
        average_reward_step_t = reward_mat.mean(dim = 0)
        J = -1 * (1 / N) * (log_prob_mat * (cumulative_reward - average_reward_step_t.reshape(1, -1))).sum()

    return J


def calculate_total_reward(trajectory_list):
    reward_mat = torch.stack([trajectory.rewards for trajectory in trajectory_list])
    return reward_mat.sum()


def get_trajectory_rewards(SAS_list):
    reward_list = []
    for s, a, s_new in SAS_list:
        reward_list.append(generate_reward(s, a, s_new))
    return reward_list