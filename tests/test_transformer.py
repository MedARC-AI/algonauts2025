import torch

from transformer import Transformer


def test_transformer():
    model = Transformer(embed_dim=256, num_heads=4, depth=4)
    print(model)

    x = torch.randn(8, 32, 256)
    x = model(x)
    assert x.shape == (8, 32, 256)
    assert not torch.any(torch.isnan(x))
