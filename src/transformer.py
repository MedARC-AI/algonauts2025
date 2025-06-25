from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp

Layer = Callable[..., nn.Module]


class Rope(nn.Module):
    freqs: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        # duplicate freqs for rotation pairs of channels
        freqs = torch.cat([freqs, freqs])
        # it's an angular freq here
        freqs = freqs * 2 * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H, C = x.shape
        coords = torch.arange(N, dtype=x.dtype, device=x.device)
        angle = coords[None, :, None, None] * self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


# from xformers
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        rope_layer: Layer = Rope,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rope = rope_layer(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.num_heads

        q = self.q(x).reshape(B, N, h, C // h).transpose(1, 2)
        k = self.k(x).reshape(B, N, h, C // h).transpose(1, 2)
        v = self.v(x).reshape(B, N, h, C // h).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int | float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Layer = nn.GELU,
        norm_layer: Layer = nn.RMSNorm,
        rope_layer: Layer = Rope,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop,
            rope_layer=rope_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Layer = nn.RMSNorm,
    ):
        super().__init__()

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[ii],
                    norm_layer=norm_layer,
                )
                for ii in range(depth)
            ],
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
