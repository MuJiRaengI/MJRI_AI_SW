import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariCNN(nn.Module):
    def __init__(
        self,
        in_channels=4,
        mid_channels=32,
        out_channels=512,
        size=(84, 84),
        norm_type="batch_norm",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.size = size
        w, h = size

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=8, stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels * 2, kernel_size=4, stride=2, padding=0
        )
        self.conv3 = nn.Conv2d(
            mid_channels * 2, mid_channels * 2, kernel_size=3, stride=1, padding=0
        )

        # Calculate output sizes after each conv layer
        # Conv1: kernel=8, stride=4, padding=0
        w1 = (w - 8) // 4 + 1
        h1 = (h - 8) // 4 + 1

        # Conv2: kernel=4, stride=2, padding=0
        w2 = (w1 - 4) // 2 + 1
        h2 = (h1 - 4) // 2 + 1

        # Conv3: kernel=3, stride=1, padding=0
        w3 = (w2 - 3) // 1 + 1
        h3 = (h2 - 3) // 1 + 1

        # Store calculated sizes for debugging
        self.feature_sizes = {
            "input": (w, h),
            "after_conv1": (w1, h1),
            "after_conv2": (w2, h2),
            "after_conv3": (w3, h3),
            "fc_input": mid_channels * 2 * h3 * w3,
        }

        # Validate minimum sizes
        if w3 <= 0 or h3 <= 0:
            raise ValueError(
                f"Input size {size} is too small for this CNN architecture. "
                f"Final conv output would be ({w3}, {h3}). "
                f"Try using a larger input size or modify the conv layers."
            )

        # Normalization layers
        if norm_type == "layer_norm":
            # LayerNorm for conv layers with calculated sizes
            self.norm1 = nn.LayerNorm([mid_channels, h1, w1])
            self.norm2 = nn.LayerNorm([mid_channels * 2, h2, w2])
            self.norm3 = nn.LayerNorm([mid_channels * 2, h3, w3])
            self.norm4 = nn.LayerNorm(out_channels)
        elif norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(mid_channels * 2)
            self.norm3 = nn.BatchNorm2d(mid_channels * 2)
            self.norm4 = nn.BatchNorm1d(out_channels)

        # Dense layer with calculated input size
        fc_input_size = mid_channels * 2 * h3 * w3
        self.fc = nn.Linear(fc_input_size, out_channels)

        # Initialize weights
        self._init_weights()

    def get_feature_sizes(self):
        """Feature map 크기 정보 반환"""
        return self.feature_sizes

    def get_architecture_info(self) -> str:
        """네트워크 구조 정보를 문자열로 반환"""
        info = []
        info.append("AtariCNN Architecture Info:")
        info.append(f"  Input size: {self.feature_sizes['input']}")
        info.append(
            f"  After Conv1 (8x8, stride=4): {self.feature_sizes['after_conv1']}"
        )
        info.append(
            f"  After Conv2 (4x4, stride=2): {self.feature_sizes['after_conv2']}"
        )
        info.append(
            f"  After Conv3 (3x3, stride=1): {self.feature_sizes['after_conv3']}"
        )
        info.append(f"  FC input features: {self.feature_sizes['fc_input']}")
        info.append(f"  Normalization type: {self.norm_type}")
        return "\n".join(info)

    def print_architecture_info(self):
        """네트워크 구조 정보 출력"""
        print(self.get_architecture_info())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.conv1(x)
        if self.norm_type == "layer_norm":
            x = self.norm1(x)
        elif self.norm_type == "batch_norm":
            x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.norm_type == "layer_norm":
            x = self.norm2(x)
        elif self.norm_type == "batch_norm":
            x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        if self.norm_type == "layer_norm":
            x = self.norm3(x)
        elif self.norm_type == "batch_norm":
            x = self.norm3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        if self.norm_type == "layer_norm":
            x = self.norm4(x)
        elif self.norm_type == "batch_norm":
            x = self.norm4(x)
        x = F.relu(x)
        return x


class QNetwork(nn.Module):
    def __init__(
        self,
        in_channels=4,
        mid_channels=32,
        out_channels=512,
        n_actions=6,
        size=(84, 84),
        norm_type="batch_norm",
        norm_input=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.n_actions = n_actions
        self.size = size
        self.norm_type = norm_type
        self.norm_input = norm_input

        # Assuming N stacked frames
        if norm_input:
            self.input_norm = nn.BatchNorm2d(in_channels)

        self.body = AtariCNN(norm_type=norm_type)
        self.final = nn.Linear(out_channels, n_actions)

    def forward(self, x):
        # Input shape: (batch, height, width, channels) -> convert to (batch, channels, height, width)
        if len(x.shape) == 4 and x.shape[-1] == 4:  # Check if channels last
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.norm_input:
            x = self.input_norm(x)
        else:
            # Normalize to [0, 1] if input is in [0, 255]
            if x.max() > 1.0:
                x = x / 255.0

        x = self.body(x)
        x = self.final(x)
        return x
