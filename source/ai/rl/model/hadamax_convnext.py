import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Sequence
from torchvision import models


class NatureCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        input_size: Tuple[int, int] = (84, 84),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size

        # ConvNeXt Tiny backbone
        self.backbone = models.convnext_tiny(pretrained=True)

        # Modify first conv to accept our input channels (4 for frame stack)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # 사전학습 모델 사용시 첫 번째 레이어만 랜덤 초기화
        nn.init.xavier_normal_(self.backbone.features[0][0].weight)
        if self.backbone.features[0][0].bias is not None:
            nn.init.constant_(self.backbone.features[0][0].bias, 0)

        # Remove the classifier head (we'll add our own)
        self.backbone.classifier = nn.Identity()

        # Get the feature size by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, *input_size)
            features = self.backbone(dummy_input)
            self.feature_size = features.shape[1]

    def forward(self, x):
        return self.backbone(x)


class QNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        in_channels: int = 4,
        norm_input: bool = False,
        input_size: Tuple[int, int] = (84, 84),
        hidden_dim: int = 512,  # Dense layer 크기 조절
    ):
        super().__init__()
        self.action_dim = action_dim
        self.norm_input = norm_input
        self.in_channels = in_channels
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Input normalization
        if norm_input:
            self.input_norm = nn.BatchNorm2d(in_channels)

        # ConvNeXt backbone
        self.nature_cnn = NatureCNN(
            in_channels=in_channels,
            input_size=input_size,
        )

        # Get feature size from NatureCNN
        feature_size = self.nature_cnn.feature_size

        # Dense layers with GELU activation (ConvNeXt style)
        self.dense = nn.Linear(feature_size, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Handle input format: (B, H, W, C) -> (B, C, H, W)
        if len(x.shape) == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.norm_input:
            x = self.input_norm(x)
        else:
            x = x / 255.0

        x = self.nature_cnn(x)
        x = nn.Flatten()(x)
        x = self.dense(x)
        x = self.final_norm(x)
        x = F.gelu(x)  # ConvNeXt uses GELU
        x = self.action_head(x)
        return x
