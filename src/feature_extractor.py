import fnmatch
from typing import Any, List, Dict

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

    def __init__(self, model: nn.Module, layers: list[str], detach: bool = True, call_fn=None):
        self.model = model
        # self.layers = layers
        self.detach = detach
        self._call_fn = call_fn

        self.layers = self._expand_layers(model, layers)

        self._features = {}
        self._handles = {}
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.layers:
            sub_module = self.model.get_submodule(layer)
            handle = sub_module.register_forward_hook(self._make_hook(layer))
            self._handles[layer] = handle

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            self._features[layer_name] = output

        return hook
    
    def clear(self):
        self._features.clear()

    def get_features(self) -> dict[str, torch.Tensor]:
        """Get the last recorded features.."""
        return self._features.copy()

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward the model and return the output and recorded features."""
        self.clear()
        output = self.model(*args, **kwargs)
        features = self.get_features()
        return output, features

    def __del__(self):
        for handle in self._handles.values():
            handle.remove()
    
    def __enter__(self):
        """Enter context: hooks are already registered."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context: remove all hooks."""
        self.remove_hooks()

    __call__ = forward

    @staticmethod
    def _expand_layers(model: nn.Module, layers: List[str]) -> List[str]:
        """
        Expand a list of layer names and/or glob patterns to all matching module names
        in the given model. Raises an error if a specified name or pattern doesn't match.
        """
        all_layers = [name for name, _ in model.named_modules() if name]  # skip the root module ''
        all_layers_set = set(all_layers)
        expanded = []
        special_chars = set("*?[]")
        for layer in layers:
            if not any(char in layer for char in special_chars):
                if layer not in all_layers_set:
                    raise ValueError(f"Layer '{layer}' not found in the model.")
                expanded.append(layer)
            else:
                matches = fnmatch.filter(all_layers, layer)
                if not matches:
                    raise ValueError(f"No layers match the pattern '{layer}'.")
                expanded.extend(matches)
        return expanded

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
