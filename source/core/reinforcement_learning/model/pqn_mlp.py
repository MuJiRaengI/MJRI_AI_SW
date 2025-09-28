import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        norm_type: str = "layer_norm",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build MLP layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization layer
            if norm_type == "layer_norm":
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm_type == "batch_norm":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norms.append(nn.Identity())

            # Dropout layer
            if dropout_rate > 0.0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())

            prev_dim = hidden_dim

    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)

        return x


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int = 4,
        hidden_dims: List[int] = [512, 512, 256],
        norm_type: str = "layer_norm",
        dropout_rate: float = 0.0,
        norm_input: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.norm_input = norm_input

        # Input normalization (for flattened inputs)
        if norm_input:
            self.input_norm = nn.LayerNorm(input_dim)

        # MLP feature extractor
        self.mlp_network = MLPNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
        )

        # Output layer for Q-values
        self.output_dim = hidden_dims[-1]
        self.q_values = nn.Linear(self.output_dim, action_dim)

    def forward(self, x):
        # Flatten input if needed (for compatibility with CNN inputs)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Input normalization
        if self.norm_input:
            x = self.input_norm(x)

        # Extract features using MLP
        features = self.mlp_network(x)

        # Compute Q-values
        q_values = self.q_values(features)

        return q_values

    def get_features(self, x):
        """Get intermediate feature representations"""
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        if self.norm_input:
            x = self.input_norm(x)

        return self.mlp_network(x)
