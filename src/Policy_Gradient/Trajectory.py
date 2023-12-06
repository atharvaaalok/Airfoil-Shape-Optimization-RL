import torch

from .Helper import *
import concurrent.futures

class Trajectory:
    def __init__(self, s0, a_params, T, policy_params, MDP_functions, parallelize = False, calculate_rewards = True):
        self.s0 = s0
        self.r0 = torch.tensor(0.0)
        self.T = T
        self.SARS = []
        self.action_log_prob = torch.zeros(T)
        self.rewards = torch.zeros(T)
        
        # Generate the trajectory
        self.generate_trajectory(a_params, policy_params, MDP_functions, parallelize, calculate_rewards)
    
    def generate_trajectory(self, a_params, policy_params, MDP_functions, parallelize, calculate_rewards):
        # Get the MDP functions
        generate_action = MDP_functions['generate_action']
        generate_next_state = MDP_functions['generate_next_state']
        generate_reward = MDP_functions['generate_reward']

        log_prob_list = []
        reward_list = []
        # Get the first state
        s = self.s0
        # Generate reward for the first state
        self.r0 = generate_reward(s, s, s)
        for t in range(self.T):
            # Generate action given the state
            a, action_log_prob = generate_action(s, a_params, policy_params)
            # Generate new state using this action
            s_new = generate_next_state(s, a)
            # Generate the reward
            if parallelize:
                r = 0
            else:
                if calculate_rewards:
                    r = torch.tensor(generate_reward(s, a, s_new))
                else:
                    r = torch.tensor(0.0)

            # Create state-action-reward tuple
            self.SARS.append(SARS_tuple(s, a, r, s_new))
            log_prob_list.append(action_log_prob)
            reward_list.append(r)

            # Update the state
            s = s_new
        
        self.action_log_prob = torch.stack(log_prob_list)
        self.rewards = torch.tensor(reward_list)
    
    def set_rewards(self, reward_list):
        for SARS, r in zip(self.SARS, reward_list):
            SARS.r = r
        self.rewards = torch.tensor(reward_list)
    
    def get_SAS_list(self):
        return [(SARS.s.detach(), SARS.a.detach(), SARS.s_new.detach()) for SARS in self.SARS]
    
    def use_delta_r(self, use = False):
        if use:
            self.rewards[1:] = torch.diff(self.rewards)
            self.rewards[0] = self.rewards[0] - self.r0



class SARS_tuple:
    def __init__(self, s, a, r, s_new):
        self.s = s
        self.a = a
        self.r = r
        self.s_new = s_new


def Generate_trajectories(s0_list, a_params, T, N, policy_params, MDP_functions, parallelize):
    trajectory_list = []
    # Generate training batch
    for i_traj in range(N):
        rewards = []
        s0 = s0_list[i_traj]
        trajectory = Trajectory(s0, a_params, T, policy_params, MDP_functions, parallelize = parallelize)
        trajectory_list.append(trajectory)

    # If running in parallel, calculate rewards for generated trajectory s, a, s_new pairs afterwards in parallel
    if parallelize:
        # Get s, a, s_new pairs to calculate corresponding rewards in parallel
        SAS_list = [trajectory.get_SAS_list() for trajectory in trajectory_list]
        
        # For all trajectories calculate rewards
        with concurrent.futures.ProcessPoolExecutor(max_workers = 60) as executor:
            reward_lists = executor.map(get_trajectory_rewards, SAS_list)
        
        # Update the trajectory rewards
        for trajectory, rewards in zip(trajectory_list, reward_lists):
            trajectory.set_rewards(rewards)
    
    return trajectory_list


def add_valid_initial_states(trajectory_list, Valid_initial_states):
    for trajectory in trajectory_list:
        for SARS in trajectory.SARS:
            if SARS.r > NEGATIVE_REWARD + 1:
                # Add the new state in the transition to the list of valid initial states
                Valid_initial_states.append(SARS.s_new)