"""
Defines encoder for the modalities of the synthetic data
"""
import torch.nn as nn


class CharModelBatchNorm(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, self.output_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # self.bn4 = nn.BatchNorm2d(self.output_dim)
        self.adaptive_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.view(int(x.size(0)), -1)
        return x


