import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NatureCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        norm_type: str = "layer_norm",
        input_size: Tuple[int, int] = (84, 84),
        channels: int = 32,  # Base channels, will be used as (channels, channels*2, channels*2)
    ):
        super().__init__()
        self.norm_type = norm_type
        self.in_channels = in_channels
        self.input_size = input_size
        self.base_channels = channels
        # Calculate channel configuration: (channels, channels*2, channels*2)
        self.channels = (channels, channels * 2, channels * 2)

        # Initialize layers based on encoder type
        self._init_baseline_layers()

    def _init_baseline_layers(self):
        # Baseline CNN layers
        conv1_ch, conv2_ch, conv3_ch = self.channels
        self.conv1 = nn.Conv2d(
            self.in_channels, conv1_ch, kernel_size=8, stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(conv1_ch, conv2_ch, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(conv2_ch, conv3_ch, kernel_size=3, stride=1, padding=0)

        # Normalization layers for baseline
        if self.norm_type == "layer_norm":
            # Will be set after calculating output sizes
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None
        elif self.norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm2d(conv1_ch)
            self.norm2 = nn.BatchNorm2d(conv2_ch)
            self.norm3 = nn.BatchNorm2d(conv3_ch)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        # Calculate sizes for LayerNorm if needed
        if self.norm_type == "layer_norm":
            # Use actual input size instead of assuming (84, 84)
            h, w = self.input_size
            h1, w1 = (h - 8) // 4 + 1, (w - 8) // 4 + 1
            h2, w2 = (h1 - 4) // 2 + 1, (w1 - 4) // 2 + 1
            h3, w3 = h2 - 3 + 1, w2 - 3 + 1

            self.norm1 = nn.LayerNorm([conv1_ch, h1, w1])
            self.norm2 = nn.LayerNorm([conv2_ch, h2, w2])
            self.norm3 = nn.LayerNorm([conv3_ch, h3, w3])

    def _get_normalize_fn(self):
        if self.norm_type == "layer_norm":
            return lambda x, norm_layer: norm_layer(x) if norm_layer else x
        elif self.norm_type == "batch_norm":
            return lambda x, norm_layer: norm_layer(x) if norm_layer else x
        else:
            return lambda x, norm_layer: x

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        return x


class QNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        in_channels: int = 4,
        norm_type: str = "layer_norm",
        norm_input: bool = False,
        input_size: Tuple[int, int] = (84, 84),
        channels: int = 32,  # Base channels, will be used as (channels, channels*2, channels*2)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.norm_input = norm_input
        self.in_channels = in_channels
        self.input_size = input_size
        self.base_channels = channels
        # Calculate channel configuration: (channels, channels*2, channels*2)
        self.channels = (channels, channels * 2, channels * 2)

        # Input normalization
        if norm_input:
            self.input_norm = nn.BatchNorm2d(in_channels)

        # CNN backbone
        self.nature_cnn = NatureCNN(
            in_channels=in_channels,
            norm_type=norm_type,
            input_size=input_size,
            channels=channels,
        )

        # Calculate feature size based on encoder type and input size
        h, w = input_size
        conv1_ch, conv2_ch, conv3_ch = self.channels
        # Baseline: conv1(stride=4) -> conv2(stride=2) -> conv3(stride=1)
        h1, w1 = (h - 8) // 4 + 1, (w - 8) // 4 + 1
        h2, w2 = (h1 - 4) // 2 + 1, (w1 - 4) // 2 + 1
        h3, w3 = h2 - 3 + 1, w2 - 3 + 1
        feature_size = conv3_ch * h3 * w3

        # Dense layers
        self.dense = nn.Linear(feature_size, 512)
        self.action_head = nn.Linear(512, action_dim)

        # Final normalization
        if norm_type == "layer_norm":
            self.final_norm = nn.LayerNorm(512)
        elif norm_type == "batch_norm":
            self.final_norm = nn.BatchNorm1d(512)
        else:
            self.final_norm = nn.Identity()

    def forward(self, x):
        # Handle input format: (B, H, W, C) -> (B, C, H, W)
        if len(x.shape) == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.norm_input:
            x = self.input_norm(x)
        else:
            x = x / 255.0

        x = self.nature_cnn(x)
        x = self.dense(x)
        x = self.final_norm(x)

        x = F.relu(x)

        x = self.action_head(x)
        return x
