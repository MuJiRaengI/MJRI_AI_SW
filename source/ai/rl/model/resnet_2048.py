import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNet2048Extractor(nn.Module):
    def __init__(
        self,
        size,
        max_exp,
        features_dim=128,
        n_blocks=3,
        channels=64,
    ):
        super().__init__()
        self.channels = channels
        self.size = size
        self.max_exp = max_exp
        self.conv_in = nn.Conv2d(self.max_exp + 1, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResNetBlock(channels) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * self.size * self.size, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # obs: (batch, 4, 4) or (batch, 1, 4, 4)
        if obs.dim() == 3:
            obs = obs.unsqueeze(1)  # (batch, 1, 4, 4)
        x = F.relu(self.bn_in(self.conv_in(obs.float())))
        x = self.res_blocks(x)
        x = self.head(x)
        return x
