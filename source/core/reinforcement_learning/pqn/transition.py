from dataclasses import dataclass

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_obs: torch.Tensor
    q_val: torch.Tensor
