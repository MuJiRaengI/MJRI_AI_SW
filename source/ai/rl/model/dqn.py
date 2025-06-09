import torch
from torch import nn


class DQNBreakout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
