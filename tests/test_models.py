import pytest
import torch

from models import MultiSubjectConvLinearEncoder


@pytest.mark.parametrize("global_pool", ["avg", "linear", "attn"])
def test_multi_subject_conv_linear_encoder(global_pool: str):
    feat_dims = [64, (8, 32), 96]
    model = MultiSubjectConvLinearEncoder(
        num_subjects=4,
        feat_dims=feat_dims,
        global_pool=global_pool,
        embed_dim=64,
        target_dim=145,
    )
    print(model)

    features = [
        torch.randn((8, 29) + ((dim,) if isinstance(dim, int) else dim))
        for dim in feat_dims
    ]

    output = model.forward(features)
    assert output.shape == (8, 4, 29, 145)
    assert not torch.any(torch.isnan(output))
