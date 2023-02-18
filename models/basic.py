import torch.nn.functional as F

from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self,
                 kernel_size_1=(9, 9),
                 kernel_size_2=(7, 7),
                 kernel_size_3=(5, 5),
                 kernel_size_4=(3, 3),
                 kernel_size_5=(3, 3),
                 stride_1=(1, 1),
                 stride_2=(1, 1),
                 stride_3=(1, 1),
                 stride_4=(1, 1),
                 stride_5=(1, 1),
                 out_channels_1=16,
                 out_channels_2=8,
                 out_channels_3=4,
                 out_channels_4=4,
                 out_channels_5=1,
                 hidden_size_1=512,
                 hidden_size_2=256,
                 ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels_1, kernel_size=kernel_size_1, stride=stride_1)
        self.pool1 = nn.MaxPool2d(kernel_size_1, stride=stride_1)

        self.conv2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size_2,
                               stride=stride_2)
        self.pool2 = nn.MaxPool2d(kernel_size_2, stride=stride_2)

        self.conv3 = nn.Conv2d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size_3,
                               stride=stride_3)
        self.pool3 = nn.MaxPool2d(kernel_size_3, stride=stride_3)

        self.conv4 = nn.Conv2d(in_channels=out_channels_3, out_channels=out_channels_4, kernel_size=kernel_size_4,
                               stride=stride_4)
        self.pool4 = nn.MaxPool2d(kernel_size_4, stride=stride_4)

        self.conv5 = nn.Conv2d(in_channels=out_channels_4, out_channels=out_channels_5, kernel_size=kernel_size_5,
                               stride=stride_5)
        self.pool5 = nn.MaxPool2d(kernel_size_5, stride=stride_5)

        in_features = 1296
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_size_1)
        self.fc2 = nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.pool5(self.conv5(x)))
        x = F.relu(self.fc1(x.flatten(1)))
        x = F.relu(self.fc2(x))
        return x
