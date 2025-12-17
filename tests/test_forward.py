import torch
from mfvit.models.mf_vit import MFViT, MFViTConfig


def test_forward_shapes():
    cfg = MFViTConfig.from_variant("mfvit-s")
    model = MFViT(cfg)
    x = torch.rand(2, 3, 512, 512)
    y = model(x)
    assert y.shape == x.shape
