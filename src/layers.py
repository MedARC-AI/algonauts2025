import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    """Conv1d layer with a causal mask, to only "attend" to past time points."""

    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        causal: bool = True,
        positive: bool = False,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        assert not causal or kernel_size % 2 == 1, "causal conv requires odd kernel"

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.causal = causal
        self.positive = positive

        attn_mask = torch.ones(kernel_size)
        if self.causal:
            attn_mask[kernel_size // 2 + 1 :] = 0.0
        self.register_buffer("attn_mask", attn_mask)

    def extra_repr(self):
        s = super().extra_repr()
        s = f"{s}, causal={self.causal}, positive={self.positive}"
        return s

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.attn_mask

        if self.positive:
            weight = weight**2

        return F.conv1d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
