import torch
import numpy as np

from torch import nn, Tensor
from models.basic import FeatureExtractor


class DQN(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.output = nn.Linear(self.feature_extractor.fc2.out_features, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output(x)


class D3QN(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.value_function = nn.Linear(self.feature_extractor.fc2.out_features, output_size)
        self.advantage = nn.Linear(self.feature_extractor.fc2.out_features, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        value = self.value_function(x)
        advantage = self.advantage(x)

        q_value = value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        return q_value
