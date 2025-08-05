import fnmatch
from typing import Any

import torch
from torch import nn


class FeatureExtractor:
    """Extract features from a torch model.

    Args:
        model: torch model
        layers: list of layer names or glob patterns
        call_fn: name of method to call instead of calling the model directly

    Example:
        extractor = FeatureExtractor(model, ['blocks.*', 'blocks.*.mlp.act'])
        output, features = extractor(input)
        feat = features['blocks.1']
    """

    def __init__(
        self,
        model: nn.Module,
        layers: list[str],
        call_fn: str | None = None,
    ):
        self.model = model
        # expand layer glob patterns
        all_layers = [name for name, _ in model.named_modules()]
        self.layers = [
            layer for pat in layers for layer in fnmatch.filter(all_layers, pat)
        ]
        self.call_fn = call_fn

        self.features = {}
        self.handles = {}

        # register forward hooks for each layer
        for layer in self.layers:
            sub_module = self.model.get_submodule(layer)
            handle = sub_module.register_forward_hook(self.make_hook(layer))
            self.handles[layer] = handle

    def make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            self.features[layer_name] = output

        return hook

    def forward(self, *args, **kwargs) -> Any:
        """Forward the model and return the original output."""
        self.features.clear()
        forward = getattr(self.model, self.call_fn) if self.call_fn else self.model
        return forward(*args, **kwargs)

    def forward_features(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Forward the model and return just the features."""
        self.features.clear()
        forward = getattr(self.model, self.call_fn) if self.call_fn else self.model
        forward(*args, **kwargs)
        return self.features.copy()

    def clear(self):
        self.features.clear()

    def __del__(self):
        for handle in self.handles.values():
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
