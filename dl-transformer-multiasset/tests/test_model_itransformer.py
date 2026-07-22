"""
Tests for model.py: ITransformer and build_model factory
"""
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from scripts.model import ITransformer, build_model, PatchTST

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def test_itransformer_forward_shape():
    """Test ITransformer forward pass produces (B,) output."""
    batch_size = 8
    lookback = 60
    n_features = 58

    model = ITransformer(
        n_features=n_features,
        lookback=lookback,
        d_model=128,
        n_heads=8,
        n_layers=3,
        dropout=0.2
    )

    x = torch.randn(batch_size, lookback, n_features)
    y = model(x)

    assert y.shape == (batch_size,), f"Expected output shape ({batch_size},), got {y.shape}"


def test_build_model_dispatch():
    """Test build_model factory returns correct class for each architecture."""
    cfg = {
        "LOOKBACK": 60,
        "D_MODEL": 64,
        "N_HEADS": 4,
        "N_LAYERS": 2,
        "DROPOUT": 0.1,
        "PATCH_LEN": 16,
        "STRIDE": 8,
    }
    n_features = 58

    # Test PatchTST dispatch
    model_patchtst = build_model("patchtst", n_features, cfg)
    assert isinstance(model_patchtst, PatchTST), f"Expected PatchTST, got {type(model_patchtst)}"

    # Test ITransformer dispatch
    model_itransformer = build_model("itransformer", n_features, cfg)
    assert isinstance(model_itransformer, ITransformer), f"Expected ITransformer, got {type(model_itransformer)}"

    # Test unknown architecture raises ValueError
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("unknown_arch", n_features, cfg)
