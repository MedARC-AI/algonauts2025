from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d layer with a causal mask, to only "attend" to past time points."""

    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str | int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        assert kernel_size % 2 == 1, "causal conv requires odd kernel size"
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

        attn_mask = torch.zeros(kernel_size)
        attn_mask[: kernel_size // 2 + 1] = 1.0
        self.weight.data.mul_(attn_mask)
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.attn_mask
        return F.conv1d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 11,
        causal: bool = False,
    ):
        super().__init__()
        conv_layer = CausalConv1d if causal else nn.Conv1d
        self.conv = conv_layer(
            in_features,
            in_features,
            kernel_size=kernel_size,
            padding="same",
            groups=in_features,
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = self.fc(x)
        return x


class LinearConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 11,
        causal: bool = False,
    ):
        super().__init__()
        conv_layer = CausalConv1d if causal else nn.Conv1d
        self.fc = nn.Linear(in_features, out_features)
        self.conv = conv_layer(
            out_features,
            out_features,
            kernel_size=kernel_size,
            padding="same",
            groups=out_features,
        )

    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        x = self.fc(x)
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        return x


class CrossSubjectConvLinearEncoder(nn.Module):
    weight: torch.Tensor

    def __init__(
        self,
        num_subjects: int = 4,
        fmri_dim: int = 1000,
        embed_dim: int = 256,
        encoder_kernel_size: int = 11,
        decoder_kernel_size: int = 11,
        normalize: bool = False,
        with_shared_encoder: bool = True,
        with_shared_decoder: bool = True,
        with_subject_encoders: bool = True,
        with_subject_decoders: bool = True,
    ):
        super().__init__()
        assert with_shared_encoder or with_subject_encoders
        assert with_shared_decoder or with_subject_decoders

        self.num_subjects = num_subjects

        if with_shared_encoder:
            self.shared_encoder = nn.Linear(fmri_dim, embed_dim)
        else:
            self.register_module("shared_encoder", None)

        if with_subject_encoders:
            if encoder_kernel_size > 1:
                encoder_fn = partial(LinearConv, kernel_size=encoder_kernel_size)
            else:
                encoder_fn = nn.Linear
            self.subject_encoders = nn.ModuleList(
                [encoder_fn(fmri_dim, embed_dim) for _ in range(num_subjects)]
            )
        else:
            self.register_module("subject_encoders", None)

        self.norm = nn.LayerNorm(embed_dim) if normalize else nn.Identity()

        if with_shared_decoder:
            self.shared_decoder = nn.Linear(embed_dim, fmri_dim)
        else:
            self.register_module("shared_decoder", None)

        if with_subject_decoders:
            if decoder_kernel_size > 1:
                decoder_fn = partial(ConvLinear, kernel_size=decoder_kernel_size)
            else:
                decoder_fn = nn.Linear
            self.subject_decoders = nn.ModuleList(
                [decoder_fn(embed_dim, fmri_dim) for _ in range(num_subjects)]
            )
        else:
            self.register_module("subject_decoders", None)

        # todo: could learn the averaging weights
        weight = (1.0 - torch.eye(self.num_subjects)) / (self.num_subjects - 1.0)
        self.register_buffer("weight", weight)
        self.apply(init_weights)

    def forward(self, input: torch.Tensor):
        # input: (N, S, L, C)
        # subject specific encoders

        if self.shared_encoder is not None:
            shared_embed = self.shared_encoder(input)
        else:
            shared_embed = 0.0

        if self.subject_encoders is not None:
            subject_embeds = torch.stack(
                [
                    encoder(input[:, ii])
                    for ii, encoder in enumerate(self.subject_encoders)
                ],
                dim=1,
            )
        else:
            subject_embeds = 0.0

        embed = self.norm(shared_embed + subject_embeds)

        # average pool the latents for all but target subject
        embed = torch.einsum("nslc,ts->ntlc", embed, self.weight)

        # subject specific decoders
        if self.shared_decoder is not None:
            shared_output = self.shared_decoder(embed)
        else:
            shared_output = 0.0

        if self.subject_decoders is not None:
            subject_outputs = torch.stack(
                [
                    decoder(embed[:, ii])
                    for ii, decoder in enumerate(self.subject_decoders)
                ],
                dim=1,
            )
        else:
            subject_outputs = 0.0
        output = shared_output + subject_outputs
        return output


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
