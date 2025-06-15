import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Resnet18(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            out_feature = self.resnet18.conv1.out_channels
            self.resnet18.conv1 = nn.Conv2d(
                in_channels, out_feature, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, out_channels)

    def forward(self, x):
        x = self.resnet18(x)
        return x
