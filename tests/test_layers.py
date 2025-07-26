import torch

from layers import AttentionPoolLatent, LinearPoolLatent


def test_attention_pool_latent():
    pool = AttentionPoolLatent(64, 133, num_heads=4)
    print(pool)

    x = torch.randn(8, 3, 133, 64)
    x = pool(x)
    assert x.shape == (8, 3, 64)
    assert not torch.any(torch.isnan(x))


def test_linear_pool_latent():
    pool = LinearPoolLatent(64, 133)
    print(pool)

    x = torch.randn(8, 3, 133, 64)
    x = pool(x)
    assert x.shape == (8, 3, 64)
    assert not torch.any(torch.isnan(x))
