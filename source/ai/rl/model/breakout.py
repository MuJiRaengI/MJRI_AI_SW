from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .resnet import Resnet18
import torch
import torch.nn as nn


class BreakoutResnet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)

        # 입력 채널 수: observation_space.shape[0]
        n_input_channels = observation_space.shape[0]

        self.resnet = Resnet18(
            in_channels=n_input_channels, out_channels=self.features_dim
        )
        print("Resnet input channels:", n_input_channels)

    def forward(self, observations):
        return self.resnet(observations)
