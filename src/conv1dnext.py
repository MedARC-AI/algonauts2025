import torch
from torch import nn
from timm.layers import DropPath, Mlp

from layers import DepthConv1d


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 11,
        causal: bool = False,
        mlp_ratio: int | float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # depthwise conv
        self.dwconv = DepthConv1d(dim, kernel_size=kernel_size, causal=causal)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise/1x1 convs, implemented with mlp
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        input = x

        x = x.transpose(-1, -2)
        x = self.dwconv(x)
        x = x.transpose(-1, -2)

        x = self.norm(x)
        x = self.mlp(x)
        x = input + self.drop_path(x)
        return x


class Conv1dNext(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        mlp_ratio: int | float = 4.0,
        kernel_size: int = 11,
        causal: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    kernel_size=kernel_size,
                    causal=causal,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                )
                for ii in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
