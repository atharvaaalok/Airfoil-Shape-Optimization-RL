import torch

class Trajectory:
    def __init__(self, s0, T, policy_params, MDP_functions, parallelize = False):
        self.s0 = s0
        self.T = T
        self.SARS = []
        self.action_log_prob = torch.zeros(T)
        self.rewards = torch.zeros(T)
        
        # Generate the trajectory
        self.generate_trajectory(policy_params, MDP_functions, parallelize)
    
    def generate_trajectory(self, policy_params, MDP_functions, parallelize):
        # Get the MDP functions
        generate_action = MDP_functions['generate_action']
        generate_next_state = MDP_functions['generate_next_state']
        generate_reward = MDP_functions['generate_reward']

        log_prob_list = []
        reward_list = []
        # Get the first state
        s = self.s0
        for t in range(self.T):
            # Generate action given the state
            a, action_log_prob = generate_action(s, policy_params)
            # Generate new state using this action
            s_new = generate_next_state(s, a)
            # Generate the reward
            if parallelize:
                r = 0
            else:
                r = generate_reward(s, a, s_new)

            # Create state-action-reward tuple
            self.SARS.append(SARS_tuple(s, a, r, s_new))
            log_prob_list.append(action_log_prob)
            reward_list.append(r)

            # Update the state
            s = s_new
        
        self.action_log_prob = torch.stack(log_prob_list)
        self.rewards = torch.tensor(reward_list)
    
    def get_total_reward(self):
        return self.rewards.sum()

    def get_total_log_prob(self):
        return self.action_log_prob.sum()
    
    def set_rewards(self, reward_list):
        for SARS, r in zip(self.SARS, reward_list):
            SARS.r = r
        self.rewards = torch.tensor(reward_list)
    
    def get_SAS_list(self):
        return [(SARS.s.detach(), SARS.a.detach(), SARS.s_new.detach()) for SARS in self.SARS]



class SARS_tuple:
    def __init__(self, s, a, r, s_new):
        self.s = s
        self.a = a
        self.r = r
        self.s_new = s_new