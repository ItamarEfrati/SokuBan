import torch


# class RolloutStorage:
#
#     def __init__(self, device):
#         self.device = device
#         self.rollout = []
#
#     def cuda(self):
#         self.use_cuda = True
#         self.states = self.states.cuda()
#         self.rewards = self.rewards.cuda()
#         self.masks = self.masks.cuda()
#         self.actions = self.actions.cuda()
#
#     def insert(self, exp):
#         self.rollout.append(exp)
#
#     def after_update(self):
#         self.states[0].copy_(self.states[-1])
#         self.masks[0].copy_(self.masks[-1])
#
#     def compute_returns(self, next_value, gamma):
#         returns = torch.zeros(self.num_steps + 1, self.num_envs, 1)
#         if self.use_cuda:
#             returns = returns.cuda()
#         returns[-1] = next_value
#         for step in reversed(range(self.num_steps)):
#             returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
#         return returns[:-1]

class RolloutStorage:
    def __init__(self, num_steps, num_envs, state_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.states = torch.zeros(num_steps + 1, num_envs, *state_shape).to(device)
        self.rewards = torch.zeros(num_steps, num_envs, 1).to(device)
        self.masks = torch.ones(num_steps + 1, num_envs, 1).to(device)
        self.actions = torch.zeros(num_steps, num_envs, 1).long().to(device)

    def insert(self, step, state, action, reward, mask):
        self.states[step+1].copy_(state)
        self.actions[step].copy_(action)
        self.rewards[step].copy_(reward)
        self.masks[step+1].copy_(mask)

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        returns = torch.zeros(self.num_steps + 1, self.num_envs, 1).to(self.device)
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]
