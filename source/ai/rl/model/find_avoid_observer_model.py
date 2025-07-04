import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from .resnet import Resnet18
from torchvision import models
import torch
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


class ResidualMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None):
        super(ResidualMLP, self).__init__()
        if hidden_features is None:
            hidden_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # projection for skip connection if needed
        if in_features != out_features:
            self.proj = nn.Linear(in_features, out_features)
        else:
            self.proj = None

    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        if self.proj is not None:
            res = self.proj(res)
        return x + res


class FindAvoidObserverExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        vector_shape = observation_space.spaces["vector"].shape  # (4,)

        in_vector_channels = vector_shape[0]

        self.direct_mlp = nn.Sequential(
            nn.Linear(in_vector_channels, features_dim // 2),
            nn.BatchNorm1d(features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim // 2),
            nn.BatchNorm1d(features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim // 2),
            nn.BatchNorm1d(features_dim // 2),
            nn.ReLU(),
            nn.Linear(features_dim // 2, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(True),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations):
        vector = observations["vector"].float()  # (B, 4)
        vec_feat = self.direct_mlp(vector)
        return self.final(vec_feat)


# 간단한 오토인코더 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(
            12, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 15, kernel_size=7, padding=3),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out, feat


class AvoidStoppedObserverExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=256,
    ):
        super().__init__(observation_space, features_dim)
        image_shape = observation_space.spaces["image"].shape  # (9, h, w)
        vector_shape = observation_space.spaces["vector"].shape  # (4,)

        in_image_channels = image_shape[0]
        in_vector_channels = vector_shape[0]

        self.resnet = models.resnet18(weights=None)
        out_feature = self.resnet.conv1.out_channels
        in_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(
            in_image_channels,
            out_feature,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-2],
        )

        self.conv2mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, features_dim),
        )

        self.direct_mlp = nn.Sequential(
            ResidualMLP(
                in_vector_channels, features_dim // 2, hidden_features=features_dim // 2
            ),
            ResidualMLP(
                features_dim // 2, features_dim // 2, hidden_features=features_dim // 2
            ),
            nn.Linear(features_dim // 2, features_dim),
        )

        self.final = nn.Sequential(
            ResidualMLP(features_dim * 2, features_dim, hidden_features=features_dim),
            ResidualMLP(features_dim, features_dim, hidden_features=features_dim),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations):
        with torch.no_grad():
            img = observations["image"].float()
            img_feat = self.resnet(img)
        img_feat = self.conv2mlp(img_feat)

        # with torch.no_grad():
        vector = observations["vector"].float()
        vec_feat = self.direct_mlp(vector)

        feat = th.cat([img_feat, vec_feat], dim=1)
        return self.final(feat)
