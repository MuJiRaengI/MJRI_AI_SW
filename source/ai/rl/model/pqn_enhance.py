import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


class ConvBlock(nn.Module):
    """Individual convolutional block with normalization and activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm_type: str = "layer_norm",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm_type = norm_type
        self.activation = activation

        # Normalization layer (will be set after calculating output size)
        self.norm = None

        # Activation function
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "elu":
            self.act = nn.ELU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def set_norm_layer(self, output_size: Tuple[int, int]):
        """Set normalization layer after calculating output size"""
        out_channels = self.conv.out_channels
        h, w = output_size

        if self.norm_type == "layer_norm":
            self.norm = nn.LayerNorm([out_channels, h, w])
        elif self.norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class AtariCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        out_channels: int = 512,
        size: Tuple[int, int] = (84, 84),
        norm_type: str = "layer_norm",
        activation: str = "relu",
        dropout: float = 0.0,
        conv_configs: List[Dict[str, Any]] = None,
        channel_multipliers: List[int] = None,
        use_residual: bool = False,
    ):
        """
        Enhanced AtariCNN with configurable conv blocks

        Args:
            in_channels: Input channels (frame stack)
            base_channels: Base number of channels
            out_channels: Final FC layer output channels
            size: Input image size (w, h)
            norm_type: Normalization type ('batch_norm', 'layer_norm')
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
            dropout: Dropout rate for conv layers
            conv_configs: Custom conv layer configurations
            channel_multipliers: Channel multipliers for each layer
            use_residual: Whether to use residual connections where possible
        """
        super().__init__()
        self.norm_type = norm_type
        self.size = size
        self.use_residual = use_residual
        w, h = size

        # Default conv configurations if not provided
        if conv_configs is None:
            conv_configs = [
                {"kernel_size": 8, "stride": 4, "padding": 0},  # Large receptive field
                {"kernel_size": 4, "stride": 2, "padding": 0},  # Medium receptive field
                {"kernel_size": 3, "stride": 1, "padding": 0},  # Fine features
            ]

        # Default channel multipliers if not provided
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 2]  # 32 -> 64 -> 64

        # Ensure we have enough multipliers
        while len(channel_multipliers) < len(conv_configs):
            channel_multipliers.append(channel_multipliers[-1])

        self.conv_configs = conv_configs
        self.channel_multipliers = channel_multipliers

        # Build conv blocks
        self.conv_blocks = nn.ModuleList()
        current_channels = in_channels
        current_w, current_h = w, h

        self.feature_sizes = {"input": (w, h)}

        for i, (config, multiplier) in enumerate(
            zip(conv_configs, channel_multipliers)
        ):
            out_ch = base_channels * multiplier

            block = ConvBlock(
                in_channels=current_channels,
                out_channels=out_ch,
                kernel_size=config["kernel_size"],
                stride=config.get("stride", 1),
                padding=config.get("padding", 0),
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
            )

            # Calculate output size
            kernel_size = config["kernel_size"]
            stride = config.get("stride", 1)
            padding = config.get("padding", 0)

            current_w = (current_w + 2 * padding - kernel_size) // stride + 1
            current_h = (current_h + 2 * padding - kernel_size) // stride + 1

            # Validate size
            if current_w <= 0 or current_h <= 0:
                raise ValueError(
                    f"Input size {size} is too small for conv layer {i+1}. "
                    f"Output would be ({current_w}, {current_h}). "
                    f"Try using larger input size or adjust conv configs."
                )

            # Set normalization layer with correct size
            block.set_norm_layer((current_h, current_w))

            self.conv_blocks.append(block)
            self.feature_sizes[f"after_conv{i+1}"] = (current_w, current_h)

            current_channels = out_ch

        # Final FC layer
        fc_input_size = current_channels * current_h * current_w
        self.feature_sizes["fc_input"] = fc_input_size

        # Multiple FC layers for better representation
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(fc_input_size, out_channels),
            ]
        )

        # Optional additional FC layers
        if out_channels > 512:
            self.fc_layers = nn.ModuleList(
                [
                    nn.Linear(fc_input_size, out_channels // 2),
                    nn.Linear(out_channels // 2, out_channels),
                ]
            )

        # Final normalization for FC layers
        if norm_type == "layer_norm":
            self.fc_norms = nn.ModuleList([nn.LayerNorm(out_channels)])
        elif norm_type == "batch_norm":
            self.fc_norms = nn.ModuleList([nn.BatchNorm1d(out_channels)])
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        if len(self.fc_layers) > 1:
            if norm_type == "layer_norm":
                self.fc_norms.insert(0, nn.LayerNorm(out_channels // 2))
            elif norm_type == "batch_norm":
                self.fc_norms.insert(0, nn.BatchNorm1d(out_channels // 2))
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}")

        # Initialize weights
        self._init_weights()

    def get_feature_sizes(self):
        """Feature map 크기 정보 반환"""
        return self.feature_sizes

    def get_architecture_info(self) -> str:
        """네트워크 구조 정보를 문자열로 반환"""
        info = []
        info.append("Enhanced AtariCNN Architecture Info:")
        info.append(f"  Input size: {self.feature_sizes['input']}")
        info.append(f"  Number of conv blocks: {len(self.conv_blocks)}")

        for i, (block, config) in enumerate(zip(self.conv_blocks, self.conv_configs)):
            info.append(f"  Conv Block {i+1}:")
            info.append(
                f"    Channels: {block.conv.in_channels} -> {block.conv.out_channels}"
            )
            info.append(
                f"    Kernel: {config['kernel_size']}, Stride: {config.get('stride', 1)}, Padding: {config.get('padding', 0)}"
            )
            info.append(f"    Output size: {self.feature_sizes[f'after_conv{i+1}']}")
            info.append(f"    Norm: {self.norm_type}, Activation: {block.activation}")

        info.append(f"  FC layers: {len(self.fc_layers)}")
        for i, fc in enumerate(self.fc_layers):
            info.append(f"    FC{i+1}: {fc.in_features} -> {fc.out_features}")

        info.append(f"  Total FC input features: {self.feature_sizes['fc_input']}")
        info.append(f"  Residual connections: {self.use_residual}")

        return "\n".join(info)

    def print_architecture_info(self):
        """네트워크 구조 정보 출력"""
        print(self.get_architecture_info())

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network"""
        # x shape: (batch, channels, height, width)

        # Pass through conv blocks
        for i, block in enumerate(self.conv_blocks):
            residual = (
                x
                if self.use_residual and i > 0 and x.shape[1] == block.conv.out_channels
                else None
            )
            x = block(x)

            # Add residual connection if possible
            if residual is not None and x.shape == residual.shape:
                x = x + residual

        # Flatten for FC layers
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        for i, (fc, norm) in enumerate(zip(self.fc_layers, self.fc_norms)):
            x = fc(x)
            x = norm(x)
            if i < len(self.fc_layers) - 1:  # Don't apply activation to final layer
                x = F.relu(x)

        return x

    @classmethod
    def create_small(
        cls,
        in_channels=4,
        out_channels=256,
        size=(84, 84),
        norm_type="layer_norm",
        activation="relu",
        dropout=0.0,
        use_residual=True,
    ):
        """Create a smaller CNN for faster training"""
        conv_configs = [
            {"kernel_size": 8, "stride": 4, "padding": 0},
            {"kernel_size": 4, "stride": 2, "padding": 0},
        ]

        return cls(
            in_channels=in_channels,
            base_channels=16,
            out_channels=out_channels,
            size=size,
            conv_configs=conv_configs,
            channel_multipliers=[1, 2],
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            use_residual=use_residual,
        )

    @classmethod
    def create_large(
        cls,
        in_channels=4,
        out_channels=1024,
        size=(84, 84),
        norm_type="layer_norm",
        activation="relu",
        dropout=0.0,
        use_residual=True,
    ):
        """Create a larger CNN for better performance"""
        conv_configs = [
            {"kernel_size": 8, "stride": 4, "padding": 0},
            {"kernel_size": 4, "stride": 2, "padding": 0},
            {"kernel_size": 3, "stride": 1, "padding": 0},
            {"kernel_size": 3, "stride": 1, "padding": 1},  # Keep size with padding
        ]

        return cls(
            in_channels=in_channels,
            base_channels=64,
            out_channels=out_channels,
            size=size,
            conv_configs=conv_configs,
            channel_multipliers=[1, 2, 4, 4],
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            use_residual=use_residual,
        )

    @classmethod
    def create_deep(
        cls,
        in_channels=4,
        out_channels=512,
        size=(84, 84),
        norm_type="layer_norm",
        activation="relu",
        dropout=0.0,
        use_residual=True,
    ):
        """Create a deeper CNN with more layers"""
        conv_configs = [
            {"kernel_size": 7, "stride": 2, "padding": 3},
            {"kernel_size": 3, "stride": 2, "padding": 1},
            {"kernel_size": 3, "stride": 2, "padding": 1},
            {"kernel_size": 3, "stride": 2, "padding": 1},
            {"kernel_size": 3, "stride": 2, "padding": 1},
        ]

        return cls(
            in_channels=in_channels,
            base_channels=32,
            out_channels=out_channels,
            size=size,
            conv_configs=conv_configs,
            channel_multipliers=[1, 1, 2, 2, 4],
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            use_residual=use_residual,
        )


class QNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        out_channels: int = 512,
        n_actions: int = 4,
        size: Tuple[int, int] = (84, 84),
        norm_type: str = "layer_norm",
        norm_input: bool = False,
        conv_configs: List[Dict[str, Any]] = None,
        channel_multipliers: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_residual: bool = True,
        cnn_type: str = "large",  # "default", "small", "large", "deep"
    ):
        """
        Enhanced Q-Network with configurable CNN backbone

        Args:
            in_channels: Input channels (frame stack)
            base_channels: Base channels
            out_channels: CNN output features
            n_actions: Number of actions
            size: Input image size (w, h)
            norm_type: Normalization type
            norm_input: Whether to normalize input
            conv_configs: Custom conv configurations
            channel_multipliers: Channel multipliers for each layer
            activation: Activation function
            dropout: Dropout rate
            use_residual: Use residual connections
            cnn_type: Predefined CNN architecture type
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.n_actions = n_actions
        self.size = size
        self.norm_type = norm_type
        self.norm_input = norm_input
        self.cnn_type = cnn_type

        # Input normalization
        if norm_input:
            self.input_norm = nn.BatchNorm2d(in_channels)

        # Create CNN backbone based on type
        if cnn_type == "small":
            self.body = AtariCNN.create_small(
                in_channels=in_channels,
                out_channels=out_channels,
                size=size,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            )
        elif cnn_type == "large":
            self.body = AtariCNN.create_large(
                in_channels=in_channels,
                out_channels=out_channels,
                size=size,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            )
        elif cnn_type == "deep":
            self.body = AtariCNN.create_deep(
                in_channels=in_channels,
                out_channels=out_channels,
                size=size,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual,
            )
        else:  # default
            self.body = AtariCNN(
                in_channels=in_channels,
                base_channels=base_channels,
                out_channels=out_channels,
                size=size,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
                conv_configs=conv_configs,
                channel_multipliers=channel_multipliers,
                use_residual=use_residual,
            )

        # Q-value head
        self.final = nn.Linear(out_channels, n_actions)

    def forward(self, x):
        # Input shape: (batch, height, width, channels) -> convert to (batch, channels, height, width)
        if (
            len(x.shape) == 4 and x.shape[-1] == self.in_channels
        ):  # Check if channels last
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

    def get_architecture_info(self) -> str:
        """Get detailed architecture information"""
        info = []
        info.append("Q-Network Architecture Info:")
        info.append(f"  CNN Type: {self.cnn_type}")
        info.append(f"  Input channels: {self.in_channels}")
        info.append(f"  Input size: {self.size}")
        info.append(f"  Output actions: {self.n_actions}")
        info.append(f"  Input normalization: {self.norm_input}")
        info.append("")
        info.append(self.body.get_architecture_info())
        return "\n".join(info)

    def print_architecture_info(self):
        """Print detailed architecture information"""
        print(self.get_architecture_info())
