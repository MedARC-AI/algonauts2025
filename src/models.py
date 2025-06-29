from functools import partial

import torch
from torch import nn

from layers import DepthConv1d


class ConvLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 11,
        causal: bool = False,
    ):
        super().__init__()
        self.conv = DepthConv1d(
            in_features,
            kernel_size=kernel_size,
            causal=causal,
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
        positive: bool = False,
        blockwise: bool = False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.conv = DepthConv1d(
            out_features,
            kernel_size=kernel_size,
            causal=causal,
            positive=positive,
            blockwise=blockwise,
        )

    def forward(self, x: torch.Tensor):
        # x: (N, L, C)
        x = self.fc(x)
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        return x


class FeatEmbed(nn.Module):
    def __init__(
        self,
        feat_dim: int = 2048,
        embed_dim: int = 256,
        kernel_size: int = 33,
        causal: bool = True,
        positive: bool = False,
        blockwise: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim) if normalize else nn.Identity()
        if kernel_size > 1:
            self.embed = LinearConv(
                feat_dim,
                embed_dim,
                kernel_size=kernel_size,
                causal=causal,
                positive=positive,
                blockwise=blockwise,
            )
        else:
            self.embed = nn.Linear(feat_dim, embed_dim)

    def forward(self, input: torch.Tensor):
        return self.embed(self.norm(input))


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

        self.apply(_init_weights)

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


class MultiSubjectConvLinearEncoder(nn.Module):
    weight: torch.Tensor

    def __init__(
        self,
        num_subjects: int = 4,
        feat_dims: tuple[int, ...] = (2048,),
        embed_dim: int = 256,
        target_dim: int = 1000,
        hidden_model: nn.Module | None = None,
        encoder_kernel_size: int = 33,
        decoder_kernel_size: int = 0,
        encoder_causal: bool = True,
        encoder_positive: bool = False,
        encoder_blockwise: bool = False,
        encoder_normalize: bool = True,
    ):
        super().__init__()
        self.num_subjects = num_subjects

        self.feat_embeds = nn.ModuleList(
            [
                FeatEmbed(
                    feat_dim,
                    embed_dim,
                    kernel_size=encoder_kernel_size,
                    causal=encoder_causal,
                    positive=encoder_positive,
                    blockwise=encoder_blockwise,
                    normalize=encoder_normalize,
                )
                for feat_dim in feat_dims
            ]
        )

        self.hidden_model = hidden_model

        if decoder_kernel_size > 1:
            decoder_linear = partial(ConvLinear, kernel_size=decoder_kernel_size)
        else:
            decoder_linear = nn.Linear

        self.shared_decoder = nn.Linear(embed_dim, target_dim)
        self.subject_decoders = nn.ModuleList(
            [decoder_linear(embed_dim, target_dim) for _ in range(num_subjects)]
        )

        self.apply(_init_weights)

    def forward(self, inputs: list[torch.Tensor]):
        # input: (N, L, D)
        # output: (N, S, L, C)
        embed = sum(
            feat_embed(input) for input, feat_embed in zip(inputs, self.feat_embeds)
        )

        if self.hidden_model is not None:
            embed = self.hidden_model(embed)

        shared_output = self.shared_decoder(embed)
        subject_output = torch.stack(
            [decoder(embed) for decoder in self.subject_decoders],
            dim=1,
        )
        output = subject_output + shared_output[:, None]
        return output


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear, DepthConv1d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
