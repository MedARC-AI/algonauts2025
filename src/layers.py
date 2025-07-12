import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthConv1d(nn.Module):
    """Depthwise conv1d.

    Args:
        causal: use a causal convolution mask.
        positive: constrain kernel to be non-negative.
        blockwise: single kernel shared across all channels.

    Shape:
        input: (*, L, C)
        output: (*, L, C)
    """

    attn_mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        causal: bool = False,
        positive: bool = False,
        blockwise: bool = False,
        bias: bool = True,
    ):
        assert not causal or kernel_size % 2 == 1, "causal conv requires odd kernel"

        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.causal = causal
        self.positive = positive
        self.blockwise = blockwise

        if blockwise:
            weight_shape = (1, 1, kernel_size)
        else:
            weight_shape = (embed_dim, 1, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter("bias", None)

        attn_mask = torch.ones(kernel_size)
        if self.causal:
            attn_mask[kernel_size // 2 + 1 :] = 0.0
        self.register_buffer("attn_mask", attn_mask)

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (*, L, C)
        # output: (*, L, C)
        *leading_dims, L, C = input.shape
        assert C == self.embed_dim

        # (*, L, C) -> (N, C, L)
        input = input.reshape(-1, L, C).transpose(1, 2)

        weight = self.weight * self.attn_mask
        if self.positive:
            weight = weight.abs()
        if self.blockwise:
            weight = weight.expand((self.embed_dim, 1, self.kernel_size))

        output = F.conv1d(
            input, weight, self.bias, padding="same", groups=self.embed_dim
        )

        output = output.transpose(1, 2)
        output = output.reshape(leading_dims + [L, C])
        return output

    def extra_repr(self):
        return (
            f"{self.embed_dim}, kernel_size={self.kernel_size}, "
            f"causal={self.causal}, positive={self.positive}, "
            f"blockwise={self.blockwise}, bias={self.bias is not None}"
        )


class LinearPoolLatent(nn.Module):
    """Learned linear pooling over a set of features.

    Shape:
        input: (*, L, C)
        output: (*, C)
    """

    def __init__(self, embed_dim: int, feat_size: int, positive: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.feat_size = feat_size
        self.positive = positive

        self.weight = nn.Parameter(torch.empty(feat_size, embed_dim))
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (*, L, C)
        # output: (*, C)
        weight = self.weight
        if self.positive:
            weight = weight.abs()
        input = torch.sum(input * weight, dim=-2)
        return input

    def extra_repr(self):
        return f"{self.embed_dim}, feat_size={self.feat_size}, positive={self.positive}"


class AttentionPoolLatent(nn.Module):
    """Learned attention-based pooling over a set of features.

    Copied from timm with some minor changes.

    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
    """

    def __init__(
        self,
        embed_dim: int,
        feat_size: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        drop: float = 0.0,
    ):
        assert embed_dim % num_heads == 0

        super().__init__()
        self.embed_dim = embed_dim
        self.feat_size = feat_size
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(torch.zeros(feat_size, embed_dim))

        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.init_weights()

    def init_weights(self):
        # todo: maybe different inits since most of our dims are small.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor):
        *leading_dims, L, C = x.shape
        x = x.reshape(-1, L, C)
        N = len(x)
        h = self.num_heads

        # position embed
        x = x + self.pos_embed.to(x.dtype)

        # fixed learned query
        # nb, timm also had a q layer, which just applied to the learned latent. this
        # is in principle unnecessary, since the query is an unconstrained parameter.
        # however maybe there is some magic learning dynamics reason. gonna try this
        # simpler version first though.
        q = self.query.expand(N, 1, C).reshape(N, 1, h, C // h).transpose(1, 2)

        # attention
        k = self.k(x).reshape(N, L, h, C // h).transpose(1, 2)
        v = self.v(x).reshape(N, L, h, C // h).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(leading_dims + [C])
        return x


class MultiAttentionPoolLatent(nn.Module):
    def __init__(
        self,
        in_dims: list[int],
        embed_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.in_dims = in_dims
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.zeros(embed_dim))
        self.kv = nn.ModuleList([nn.Linear(dim, 2 * embed_dim) for dim in in_dims])
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: list[torch.Tensor]):
        N = x[0].shape[0]
        C = self.embed_dim
        h = self.num_heads

        # fixed learned query
        q = self.query.expand(N, 1, C)
        q = q.reshape(N, 1, h, C // h).transpose(1, 2)  # [N, h, 1, C]

        # keys, values for each input feature map
        kv = torch.cat([self.kv[ii](xi) for ii, xi in enumerate(x)], dim=1)
        _, L, _ = kv.shape
        kv = kv.reshape(N, L, 2, h, C // h).permute(2, 0, 3, 1, 4)  # [2, N, h, L, C]
        k, v = torch.unbind(kv, dim=0)

        x = F.scaled_dot_product_attention(q, k, v)  # [N, h, 1, C]
        x = x.reshape(N, C)
        return x
