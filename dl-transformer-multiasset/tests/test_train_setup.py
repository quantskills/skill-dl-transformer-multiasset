"""
Tests for train.py setup: device selection, CPU guard, mixed precision, loss functions, DEFAULT_CFG
"""
import os
import pytest
import torch
import torch.nn as nn
from scripts.train import (
    pick_device,
    guard_cpu_training,
    mixed_precision_context,
    rank_ic_loss,
    combined_loss,
    DEFAULT_CFG
)


class TestDeviceSelection:
    """Test device selection logic"""

    def test_pick_device_cpu_forced(self):
        """Test forcing CPU device"""
        device = pick_device(preference="cpu")
        assert device.type == "cpu"

    def test_pick_device_auto(self):
        """Test auto device selection (should return cuda > mps > cpu)"""
        device = pick_device(preference="auto")
        assert device.type in ["cuda", "mps", "cpu"]
        # Priority check: if cuda available, should pick cuda
        if torch.cuda.is_available():
            assert device.type == "cuda"
        # If cuda not available but mps is, should pick mps
        elif torch.backends.mps.is_available():
            assert device.type == "mps"
        # Otherwise cpu
        else:
            assert device.type == "cpu"


class TestCPUGuard:
    """Test CPU training guard"""

    def test_guard_cpu_blocks_large(self):
        """Test that CPU guard blocks large datasets"""
        device = torch.device("cpu")
        # Remove env var if exists
        old_val = os.environ.pop("ALLOW_CPU_TRAIN", None)
        try:
            with pytest.raises(RuntimeError, match="Training on CPU with .* samples may take hours"):
                guard_cpu_training(device, n_samples=15000, threshold=10000)
        finally:
            # Restore env var
            if old_val is not None:
                os.environ["ALLOW_CPU_TRAIN"] = old_val

    def test_guard_cpu_allows_override(self):
        """Test that ALLOW_CPU_TRAIN=1 overrides the guard"""
        device = torch.device("cpu")
        old_val = os.environ.get("ALLOW_CPU_TRAIN")
        try:
            os.environ["ALLOW_CPU_TRAIN"] = "1"
            # Should not raise
            guard_cpu_training(device, n_samples=15000, threshold=10000)
        finally:
            # Restore env var
            if old_val is not None:
                os.environ["ALLOW_CPU_TRAIN"] = old_val
            else:
                os.environ.pop("ALLOW_CPU_TRAIN", None)

    def test_guard_cpu_allows_small(self):
        """Test that small datasets are allowed on CPU"""
        device = torch.device("cpu")
        # Should not raise
        guard_cpu_training(device, n_samples=5000, threshold=10000)


class TestLossFunctions:
    """Test loss functions"""

    def test_combined_loss_gradient(self):
        """Test that combined loss produces valid gradients"""
        # Create dummy predictions and targets
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randn(32, 1)

        # Compute loss
        loss = combined_loss(pred, target, alpha=0.5)

        # Check loss is scalar
        assert loss.dim() == 0

        # Check loss is finite
        assert torch.isfinite(loss).item()

        # Check gradient can be computed
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestDefaultConfig:
    """Test DEFAULT_CFG contains all required keys"""

    def test_default_cfg(self):
        """Test DEFAULT_CFG has all required hyperparameters"""
        required_keys = [
            "LOOKBACK",
            "PATCH_LEN",
            "STRIDE",
            "D_MODEL",
            "N_HEADS",
            "N_LAYERS",
            "DROPOUT",
            "LR",
            "BATCH_SIZE",
            "MAX_EPOCHS",
            "EARLY_STOP_PATIENCE",
            "SEED",
            "LOSS_ALPHA"
        ]

        for key in required_keys:
            assert key in DEFAULT_CFG, f"DEFAULT_CFG missing required key: {key}"

        # Check types and reasonable values
        assert isinstance(DEFAULT_CFG["LOOKBACK"], int) and DEFAULT_CFG["LOOKBACK"] > 0
        assert isinstance(DEFAULT_CFG["PATCH_LEN"], int) and DEFAULT_CFG["PATCH_LEN"] > 0
        assert isinstance(DEFAULT_CFG["STRIDE"], int) and DEFAULT_CFG["STRIDE"] > 0
        assert isinstance(DEFAULT_CFG["D_MODEL"], int) and DEFAULT_CFG["D_MODEL"] > 0
        assert isinstance(DEFAULT_CFG["N_HEADS"], int) and DEFAULT_CFG["N_HEADS"] > 0
        assert isinstance(DEFAULT_CFG["N_LAYERS"], int) and DEFAULT_CFG["N_LAYERS"] > 0
        assert isinstance(DEFAULT_CFG["DROPOUT"], float) and 0 <= DEFAULT_CFG["DROPOUT"] < 1
        assert isinstance(DEFAULT_CFG["LR"], float) and DEFAULT_CFG["LR"] > 0
        assert isinstance(DEFAULT_CFG["BATCH_SIZE"], int) and DEFAULT_CFG["BATCH_SIZE"] > 0
        assert isinstance(DEFAULT_CFG["MAX_EPOCHS"], int) and DEFAULT_CFG["MAX_EPOCHS"] > 0
        assert isinstance(DEFAULT_CFG["EARLY_STOP_PATIENCE"], int) and DEFAULT_CFG["EARLY_STOP_PATIENCE"] > 0
        assert isinstance(DEFAULT_CFG["SEED"], int)
        assert isinstance(DEFAULT_CFG["LOSS_ALPHA"], float) and 0 <= DEFAULT_CFG["LOSS_ALPHA"] <= 1
