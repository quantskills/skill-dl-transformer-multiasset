"""
Tests for model.py: RotaryEmbedding, apply_rope, PatchTST
"""
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from scripts.model import RotaryEmbedding, apply_rope, PatchTST

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def test_rope_shapes():
    """Test RoPE produces correct (cos, sin) shapes."""
    dim = 64
    seq_len = 20
    rope = RotaryEmbedding(dim)
    cos, sin = rope(seq_len)

    assert cos.shape == (seq_len, dim), f"Expected cos shape ({seq_len}, {dim}), got {cos.shape}"
    assert sin.shape == (seq_len, dim), f"Expected sin shape ({seq_len}, {dim}), got {sin.shape}"

    # Test apply_rope with dummy tensors
    batch_size = 4
    n_heads = 8
    q = torch.randn(batch_size, n_heads, seq_len, dim)
    k = torch.randn(batch_size, n_heads, seq_len, dim)

    q_rope, k_rope = apply_rope(q, k, cos, sin)

    assert q_rope.shape == q.shape, f"Expected q_rope shape {q.shape}, got {q_rope.shape}"
    assert k_rope.shape == k.shape, f"Expected k_rope shape {k.shape}, got {k_rope.shape}"


def test_patchtst_forward_shape():
    """Test PatchTST forward pass produces (B,) output."""
    batch_size = 8
    lookback = 60
    n_features = 58

    model = PatchTST(
        n_features=n_features,
        lookback=lookback,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )

    x = torch.randn(batch_size, lookback, n_features)
    y = model(x)

    assert y.shape == (batch_size,), f"Expected output shape ({batch_size},), got {y.shape}"


def test_patchtst_backward():
    """Test PatchTST backward pass: gradients flow correctly."""
    batch_size = 4
    lookback = 60
    n_features = 58

    model = PatchTST(
        n_features=n_features,
        lookback=lookback,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )

    x = torch.randn(batch_size, lookback, n_features, requires_grad=True)
    target = torch.randn(batch_size)

    # Forward
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, target)

    # Backward
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "Input gradients should exist"
    assert x.grad.shape == x.shape, f"Expected grad shape {x.shape}, got {x.grad.shape}"

    # Check model parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "At least one model parameter should have gradients"
