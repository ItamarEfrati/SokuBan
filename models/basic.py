import torch.nn.functional as F

from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self,
                 kernel_size_1=(8, 8),
                 kernel_size_2=(4, 4),
                 kernel_size_3=(3, 3),
                 out_channels_1=32,
                 out_channels_2=64,
                 out_channels_3=64,
                 hidden_size_1=512,
                 ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=out_channels_1,
                               kernel_size=kernel_size_1,
                               stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=out_channels_1,
                               out_channels=out_channels_2,
                               kernel_size=kernel_size_2,
                               stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=out_channels_2,
                               out_channels=out_channels_3,
                               kernel_size=kernel_size_3)

        in_features = 2304
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_size_1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return x
