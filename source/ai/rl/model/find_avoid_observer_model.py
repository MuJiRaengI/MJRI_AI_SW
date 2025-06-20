import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from .resnet import Resnet18
import gym


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x + res


class FindAvoidObserverExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        image_shape = observation_space.spaces["image"].shape  # (9, h, w)
        vector_shape = observation_space.spaces["vector"].shape  # (4,)

        in_image_channels = image_shape[0]
        in_vector_channels = vector_shape[0]

        self.resnet_scene = Resnet18(in_image_channels, features_dim)
        self.conv_scene = nn.Sequential(
            nn.Conv2d(
                in_image_channels, features_dim, kernel_size=7, padding=3, stride=2
            ),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(True),
            ConvBlock(features_dim, features_dim, kernel_size=3, padding=1, stride=2),
            ResidualBlock(features_dim, kernel_size=3, padding=1),
            ConvBlock(features_dim, features_dim, kernel_size=3, padding=1, stride=2),
            ResidualBlock(features_dim, kernel_size=3, padding=1),
            ConvBlock(features_dim, features_dim, kernel_size=3, padding=1, stride=2),
            ResidualBlock(features_dim, kernel_size=3, padding=1),
            ConvBlock(features_dim, features_dim, kernel_size=3, padding=1, stride=2),
            ResidualBlock(features_dim, kernel_size=3, padding=1),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.direct_mlp = nn.Sequential(
            nn.Linear(in_vector_channels, features_dim),
        )

        self.final = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(True),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations):
        image = observations["image"].float()  # (B, 9, h, w)
        vector = observations["vector"].float()  # (B, 4)
        img_feat = self.conv_scene(image)
        vec_feat = self.direct_mlp(vector)
        concat = th.cat([img_feat, vec_feat], dim=1)
        return self.final(concat)


class FindAvoidObserverPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FindAvoidObserverExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            **kwargs,
        )
