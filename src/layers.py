import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthConv1d(nn.Module):
    """Depthwise conv1d.

    Args:
        causal: use a causal convolution mask.
        positive: constrain kernel to be non-negative.
        blockwise: single kernel shared across all channels.
    """

    attn_mask: torch.Tensor

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool = False,
        positive: bool = False,
        blockwise: bool = False,
        bias: bool = True,
    ):
        assert not causal or kernel_size % 2 == 1, "causal conv requires odd kernel"

        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.positive = positive
        self.blockwise = blockwise

        if blockwise:
            weight_shape = (1, 1, kernel_size)
        else:
            weight_shape = (channels, 1, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.empty(channels))
        else:
            self.register_parameter("bias", None)

        attn_mask = torch.ones(kernel_size)
        if self.causal:
            attn_mask[kernel_size // 2 + 1 :] = 0.0
        self.register_buffer("attn_mask", attn_mask)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.attn_mask
        if self.positive:
            weight = weight**2
        if self.blockwise:
            weight = weight.expand((self.channels, 1, self.kernel_size))

        return F.conv1d(input, weight, self.bias, padding="same", groups=self.channels)

    def extra_repr(self):
        return (
            f"{self.channels}, kernel_size={self.kernel_size}, "
            f"causal={self.causal}, positive={self.positive}, "
            f"blockwise={self.blockwise}, bias={self.bias is not None}"
        )
