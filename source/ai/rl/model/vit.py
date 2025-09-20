# vit.py
# Minimal yet solid Vision Transformer implementation (PyTorch)
# Author: you + ChatGPT
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utils
# ----------------------------
def _trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask


# ----------------------------
# Patch Embedding
# ----------------------------
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using Conv2d with kernel=stride=patch_size.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid[0] * self.grid[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, E, Gh, Gw) -> (B, N, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ----------------------------
# Attention
# ----------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Standard MHA; optionally uses SDPA for speed on PyTorch 2.x.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # correct order: q, k, v
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)

        if self.use_sdpa:
            # scaled_dot_product_attention handles scaling & dropout
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )  # (B, H, N, D)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (B,H,N,N)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v  # (B, H, N, D)

        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ----------------------------
# MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ----------------------------
# Transformer Block (Pre-LN)
# ----------------------------
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, qkv_bias, attn_drop, proj_drop=drop, use_sdpa=use_sdpa
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------
# Vision Transformer
# ----------------------------
@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    pooling: str = "cls"  # "cls" or "mean"
    use_sdpa: bool = True  # use PyTorch SDPA


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.pooling = cfg.pooling

        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # class token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

        # stochastic depth schedule
        dpr = torch.linspace(0, cfg.drop_path_rate, steps=cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    qkv_bias=True,
                    drop=cfg.drop_rate,
                    attn_drop=cfg.attn_drop_rate,
                    drop_path=dpr[i],
                    use_sdpa=cfg.use_sdpa,
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

        # classification head
        self.head = (
            nn.Linear(cfg.embed_dim, cfg.num_classes)
            if cfg.num_classes > 0
            else nn.Identity()
        )

        self._reset_parameters()

    # weight init (follows common ViT init)
    def _reset_parameters(self):
        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    def _init(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _interpolate_pos_encoding(
        self, x: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        If input resolution differs from training, interpolate positional embeddings.
        x: (B, 1+N, C), H,W: patch grid
        """
        N = x.shape[1] - 1
        N0 = self.pos_embed.shape[1] - 1
        if N == N0:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        gh0 = gw0 = int(math.sqrt(N0))
        patch_pos = patch_pos.reshape(1, gh0, gw0, -1).permute(
            0, 3, 1, 2
        )  # (1,C,gh0,gw0)
        patch_pos = F.interpolate(
            patch_pos, size=(H, W), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # tokens
        B, _, H, W = x.shape
        x = self.patch_embed(x)  # (B, N, C)
        Gh = H // self.cfg.patch_size
        Gw = W // self.cfg.patch_size

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, C)

        pos = self._interpolate_pos_encoding(x, Gh, Gw)
        x = x + pos
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, 1+N, C)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.pooling == "cls":
            x = x[:, 0]
        else:  # mean over patch tokens
            x = x[:, 1:].mean(dim=1)
        x = self.head(x)
        return x


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # config like ViT-B/16
    cfg = ViTConfig(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pooling="cls",
        use_sdpa=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViT(cfg).to(device)

    x = torch.randn(8, 3, 224, 224, device=device)
    y = model(x)  # (8, 10)
    print("Logits:", y.shape)

    # quick train step demo
    if cfg.num_classes > 0:
        target = torch.randint(0, cfg.num_classes, (8,), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
        loss = F.cross_entropy(y, target, label_smoothing=0.1)
        loss.backward()
        opt.step()
        print("Loss:", float(loss))
