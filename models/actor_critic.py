import torch
from torch import nn
import torch.nn.functional as F

from models.basic import FeatureExtractor


class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.critic = nn.Linear(self.feature_extractor.fc1.out_features, 1)
        self.actor = nn.Linear(self.feature_extractor.fc1.out_features, num_actions)

    def forward(self, x):
        x = self.feature_extractor(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def get_critic(self, x):
        x = self.feature_extractor(x)
        value = self.critic(x)
        return value

    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)

        probs = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()

        return logit, action_log_probs, value, entropy
