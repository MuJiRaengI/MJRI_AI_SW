import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Sequence


class NatureCNN(nn.Module):
    def __init__(
        self,
        encoder_type: str = "hadamax",  # "baseline" or "hadamax"
        in_channels: int = 4,
        norm_type: str = "layer_norm",
        input_size: Tuple[int, int] = (84, 84),
        channels: int = 32,  # Base channels, will be used as (channels, channels*2, channels*2)
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.norm_type = norm_type
        self.in_channels = in_channels
        self.input_size = input_size
        self.base_channels = channels
        # Calculate channel configuration: (channels, channels*2, channels*2)
        self.channels = (channels, channels * 2, channels * 2)

        # Initialize layers based on encoder type
        if encoder_type == "baseline":
            self._init_baseline_layers()
        elif encoder_type == "hadamax":
            self._init_hadamax_layers()
        else:
            raise ValueError("Invalid encoder architecture specified.")

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

    def _init_hadamax_layers(self):
        # Hadamax dual-branch layers
        conv1_ch, conv2_ch, conv3_ch = self.channels

        # First block
        self.conv1a = nn.Conv2d(
            self.in_channels, conv1_ch, kernel_size=9, stride=1, padding=4
        )  # SAME padding
        self.conv1b = nn.Conv2d(
            self.in_channels, conv1_ch, kernel_size=9, stride=1, padding=4
        )
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # Second block
        self.conv2a = nn.Conv2d(
            conv1_ch, conv2_ch, kernel_size=5, stride=1, padding=2
        )  # SAME padding
        self.conv2b = nn.Conv2d(conv1_ch, conv2_ch, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third block
        self.conv3a = nn.Conv2d(
            conv2_ch, conv3_ch, kernel_size=3, stride=1, padding=1
        )  # SAME padding
        self.conv3b = nn.Conv2d(conv2_ch, conv3_ch, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Normalization layers for hadamax
        if self.norm_type == "layer_norm":
            # Calculate sizes using actual input size
            # In hadamax, conv layers use SAME padding, so size changes only at pooling
            h, w = self.input_size
            # After conv1a/conv1b (SAME padding, stride=1): size unchanged
            # After pool1 (stride=4): size = h//4, w//4
            h1, w1 = h // 4, w // 4
            # After conv2a/conv2b (SAME padding, stride=1): size unchanged
            # After pool2 (stride=2): size = h1//2, w1//2
            h2, w2 = h1 // 2, w1 // 2
            # After conv3a/conv3b (SAME padding, stride=1): size unchanged
            # After pool3 (stride=1, padding=1): size unchanged
            h3, w3 = h2, w2

            self.norm1a = nn.LayerNorm(
                [conv1_ch, h, w]
            )  # Before pooling, size unchanged
            self.norm1b = nn.LayerNorm([conv1_ch, h, w])
            self.norm2a = nn.LayerNorm([conv2_ch, h1, w1])  # After pool1
            self.norm2b = nn.LayerNorm([conv2_ch, h1, w1])
            self.norm3a = nn.LayerNorm([conv3_ch, h2, w2])  # After pool2
            self.norm3b = nn.LayerNorm([conv3_ch, h2, w2])
        elif self.norm_type == "batch_norm":
            self.norm1a = nn.BatchNorm2d(conv1_ch)
            self.norm1b = nn.BatchNorm2d(conv1_ch)
            self.norm2a = nn.BatchNorm2d(conv2_ch)
            self.norm2b = nn.BatchNorm2d(conv2_ch)
            self.norm3a = nn.BatchNorm2d(conv3_ch)
            self.norm3b = nn.BatchNorm2d(conv3_ch)
        else:
            self.norm1a = nn.Identity()
            self.norm1b = nn.Identity()
            self.norm2a = nn.Identity()
            self.norm2b = nn.Identity()
            self.norm3a = nn.Identity()
            self.norm3b = nn.Identity()

        # Initialize weights with Xavier normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_normalize_fn(self):
        if self.norm_type == "layer_norm":
            return lambda x, norm_layer: norm_layer(x) if norm_layer else x
        elif self.norm_type == "batch_norm":
            return lambda x, norm_layer: norm_layer(x) if norm_layer else x
        else:
            return lambda x, norm_layer: x

    def forward(self, x):
        if self.encoder_type == "baseline":
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

        elif self.encoder_type == "hadamax":
            # First block
            x1 = self.conv1a(x)
            x2 = self.conv1b(x)
            x1 = self.norm1a(x1)
            x2 = self.norm1b(x2)
            x1 = F.gelu(x1)
            x2 = F.gelu(x2)
            x = x1 * x2  # Element-wise multiplication
            x = self.pool1(x)

            # Second block
            x1 = self.conv2a(x)
            x2 = self.conv2b(x)
            x1 = self.norm2a(x1)
            x2 = self.norm2b(x2)
            x1 = F.gelu(x1)
            x2 = F.gelu(x2)
            x = x1 * x2  # Element-wise multiplication
            x = self.pool2(x)

            # Third block
            x1 = self.conv3a(x)
            x2 = self.conv3b(x)
            x1 = self.norm3a(x1)
            x2 = self.norm3b(x2)
            x1 = F.gelu(x1)
            x2 = F.gelu(x2)
            x = x1 * x2  # Element-wise multiplication
            x = self.pool3(x)

            x = x.reshape(x.shape[0], -1)

        return x


class QNetwork(nn.Module):
    def __init__(
        self,
        encoder_type: str = "hadamax",  # "baseline" or "hadamax"
        action_dim: int = 4,
        in_channels: int = 4,
        norm_type: str = "layer_norm",
        norm_input: bool = False,
        input_size: Tuple[int, int] = (84, 84),
        channels: int = 32,  # Base channels, will be used as (channels, channels*2, channels*2)
    ):
        super().__init__()
        self.encoder_type = encoder_type
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
            encoder_type=encoder_type,
            in_channels=in_channels,
            norm_type=norm_type,
            input_size=input_size,
            channels=channels,
        )

        # Calculate feature size based on encoder type and input size
        h, w = input_size
        conv1_ch, conv2_ch, conv3_ch = self.channels
        if encoder_type == "baseline":
            # Baseline: conv1(stride=4) -> conv2(stride=2) -> conv3(stride=1)
            h1, w1 = (h - 8) // 4 + 1, (w - 8) // 4 + 1
            h2, w2 = (h1 - 4) // 2 + 1, (w1 - 4) // 2 + 1
            h3, w3 = h2 - 3 + 1, w2 - 3 + 1
            feature_size = conv3_ch * h3 * w3
        else:  # hadamax
            # Hadamax: pool1(stride=4) -> pool2(stride=2) -> pool3(stride=1)
            h1, w1 = h // 4, w // 4
            h2, w2 = h1 // 2, w1 // 2
            h3, w3 = h2, w2  # stride=1 preserves size
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

        # # Initialize weights
        # nn.init.he_normal_(self.dense.weight)
        # nn.init.he_normal_(self.action_head.weight)
        # if self.dense.bias is not None:
        #     nn.init.constant_(self.dense.bias, 0)
        # if self.action_head.bias is not None:
        #     nn.init.constant_(self.action_head.bias, 0)

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

        if self.encoder_type == "hadamax":
            x = F.gelu(x)
        else:
            x = F.relu(x)

        x = self.action_head(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            4 if channels == 16 else channels // 2, channels, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(channels)
        self.res2 = ResidualBlock(channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class QNetwork_Impala(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        in_channels: int = 4,
        norm_type: str = "layer_norm",
        norm_input: bool = False,
        channels: Sequence[int] = (16, 32, 32, 64, 64),
        input_size: Tuple[int, int] = (84, 84),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.norm_input = norm_input
        self.channels = channels
        self.in_channels = in_channels
        self.input_size = input_size

        # Input normalization
        if norm_input:
            self.input_norm = nn.BatchNorm2d(in_channels)

        # Conv sequences
        self.conv_sequences = nn.ModuleList()
        current_channels = in_channels
        for ch in channels:
            conv_seq = ConvSequence(ch)
            # Manually set input channels for first layer
            conv_seq.conv = nn.Conv2d(current_channels, ch, kernel_size=3, padding=1)
            self.conv_sequences.append(conv_seq)
            current_channels = ch

        # Dense layers
        # Calculate feature size more accurately
        h, w = input_size
        # Each conv sequence has stride=2 pooling
        for _ in channels:
            h = (h + 1) // 2  # ceiling division for stride=2 with padding
            w = (w + 1) // 2
        feature_size = channels[-1] * h * w
        self.dense1 = nn.Linear(feature_size, 256)
        self.action_head = nn.Linear(256, action_dim)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.he_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle input format: (B, H, W, C) -> (B, C, H, W)
        if len(x.shape) == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.norm_input:
            x = self.input_norm(x)
        else:
            x = x / 255.0

        for conv_seq in self.conv_sequences:
            x = conv_seq(x)

        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.action_head(x)
        return x
