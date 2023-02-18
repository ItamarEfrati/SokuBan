import torch
from torch import nn
import torch.nn.functional as F

from models.basic import FeatureExtractor


class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()

        # self.normalization = normalization  # whether normalize pre-trained value functions
        # self.steps_done = 0

        self.feature_extractor = FeatureExtractor()
        self.critic = nn.Linear(self.feature_extractor.fc2.out_features, 1)
        self.actor = nn.Linear(self.feature_extractor.fc2.out_features, num_actions)

    def forward(self, x):
        x = self.feature_extractor(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def get_critic(self, x):
        x = self.feature_extractor(x)
        value = self.critic(x)
        # if self.normalization:
        #     value = value / torch.sum(value)
        return value
