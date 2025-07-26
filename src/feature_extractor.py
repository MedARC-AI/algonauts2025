import fnmatch
from typing import Any

import torch
from torch import nn


class FeatureExtractor:
    """Extract features from a torch model.

    Args:
        model: torch model
        layers: list of layer names or glob patterns

    Example:
        extractor = FeatureExtractor(model, ['blocks.*', 'blocks.*.mlp.act'])
        output, features = extractor(input)

        one_feat = features['blocks.1']
    """

    def __init__(self, model: nn.Module, layers: list[str]):
        self.model = model
        self.layers = layers

        all_layers = [name for name, _ in model.named_modules()]
        self.expanded_layers = [
            layer for pat in layers for layer in fnmatch.filter(all_layers, pat)
        ]

        self._features = {}
        self._handles = {}

        # register forward hooks for each layer
        for layer in self.expanded_layers:
            sub_module = self.model.get_submodule(layer)
            handle = sub_module.register_forward_hook(self._make_hook(layer))
            self._handles[layer] = handle

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            self._features[layer_name] = output

        return hook

    def get_features(self) -> dict[str, torch.Tensor]:
        """Get the last recorded features.."""
        return self._features.copy()

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward the model and return the output and recorded features."""
        self._features.clear()
        output = self.model(*args, **kwargs)
        features = self.get_features()
        return output, features

    def __del__(self):
        for handle in self._handles.values():
            handle.remove()

    __call__ = forward


class FeatureAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(
        self,
        output_size: int | tuple[int, int] = 8,
        num_prefix_tokens: int = 0,
    ):
        super().__init__(output_size)
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.num_prefix_tokens :]

        N, L, C = x.shape
        H = W = int(L**0.5)
        assert H * W == L

        x = x.transpose(1, 2).reshape(N, C, H, W)
        x = super().forward(x)

        x = x.flatten(2).transpose(1, 2)
        return x
