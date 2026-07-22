"""
Tests for walk-forward training loop: _SeqDataset, make_folds, train_one_fold
"""
import os
import pytest
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scripts.train import (
    _SeqDataset,
    make_folds,
    train_one_fold,
    DEFAULT_CFG
)


class TestSeqDataset:
    """Test _SeqDataset builds (x[L,F], y) windows per symbol"""

    def test_seq_dataset_shape(self):
        """Test that _SeqDataset produces correct shapes"""
        # Create synthetic feature dataframe
        # 3 symbols, 100 days each, 10 features + label
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        symbols = ["AG", "AL", "AU"]

        rows = []
        for sym in symbols:
            for date in dates:
                row = {"date": date, "underlying_symbol": sym}
                # 10 features
                for i in range(10):
                    row[f"feature_{i}"] = np.random.randn()
                # 1 label
                row["label_5d_rank"] = np.random.rand()
                rows.append(row)

        df = pd.DataFrame(rows)

        # Create dataset with lookback=20
        lookback = 20
        dataset = _SeqDataset(df, lookback=lookback)

        # Check dataset length: 3 symbols * (100 - 20) = 240
        assert len(dataset) == 3 * (100 - lookback)

        # Check sample shape
        x, y = dataset[0]
        assert x.shape == (lookback, 10), f"Expected x.shape=(20, 10), got {x.shape}"
        assert y.shape == (1,), f"Expected y.shape=(1,), got {y.shape}"

        # Check all samples have correct shape
        for i in range(len(dataset)):
            x, y = dataset[i]
            assert x.shape == (lookback, 10)
            assert y.shape == (1,)


class TestMakeFolds:
    """Test make_folds generates 5 folds with correct splits"""

    def test_make_folds_shape(self):
        """Test make_folds returns 5 folds with increasing train sizes"""
        # Create synthetic feature dataframe spanning 2015-2028 (14 years)
        # to accommodate all 5 folds with test data
        dates = pd.date_range("2015-01-01", "2028-12-31", freq="D")
        symbols = ["AG", "AL"]

        rows = []
        for sym in symbols:
            for date in dates:
                row = {"date": date, "underlying_symbol": sym}
                # 5 features
                for i in range(5):
                    row[f"feature_{i}"] = np.random.randn()
                # 1 label
                row["label_5d_rank"] = np.random.rand()
                rows.append(row)

        df = pd.DataFrame(rows)

        # Generate 5 folds: train=6y, val=1y, test=1y, step=1y, start=2015
        folds = make_folds(
            df,
            train_years=6,
            val_years=1,
            test_years=1,
            step=1,
            start=2015
        )

        # Should have 5 folds
        assert len(folds) == 5

        # Check each fold structure
        for i, fold in enumerate(folds):
            assert "fold_id" in fold
            assert "train" in fold
            assert "val" in fold
            assert "test" in fold
            assert fold["fold_id"] == i

            # Check train/val/test are DataFrames
            assert isinstance(fold["train"], pd.DataFrame)
            assert isinstance(fold["val"], pd.DataFrame)
            assert isinstance(fold["test"], pd.DataFrame)

            # Check train size increases
            assert len(fold["train"]) > 0
            assert len(fold["val"]) > 0
            # For last few folds, test may be empty if data doesn't extend far enough
            # This is expected behavior

        # Check train sizes are increasing
        train_sizes = [len(fold["train"]) for fold in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], \
                f"Train size should not decrease: fold {i-1} has {train_sizes[i-1]}, fold {i} has {train_sizes[i]}"


class TestTrainOneFold:
    """Test train_one_fold smoke test"""

    def test_train_one_fold_smoke(self, tmp_path):
        """Smoke test: 1 epoch on synthetic data, should complete without error"""
        # Create minimal synthetic data
        dates = pd.date_range("2020-01-01", periods=120, freq="D")
        symbols = ["AG", "AL"]

        rows = []
        for sym in symbols:
            for date in dates:
                row = {"date": date, "underlying_symbol": sym}
                # 8 features
                for i in range(8):
                    row[f"feature_{i}"] = np.random.randn()
                # 1 label
                row["label_5d_rank"] = np.random.rand()
                rows.append(row)

        df_all = pd.DataFrame(rows)

        # Split into train/val
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]

        df_train = df_all[df_all["date"].isin(train_dates)].copy()
        df_val = df_all[df_all["date"].isin(val_dates)].copy()

        fold = {
            "fold_id": 0,
            "train": df_train,
            "val": df_val,
            "test": df_val  # Reuse val for test in smoke test
        }

        # Override config for fast smoke test
        cfg = DEFAULT_CFG.copy()
        cfg["LOOKBACK"] = 20
        cfg["PATCH_LEN"] = 10
        cfg["STRIDE"] = 5
        cfg["D_MODEL"] = 32
        cfg["N_HEADS"] = 4
        cfg["N_LAYERS"] = 2
        cfg["DROPOUT"] = 0.1
        cfg["LR"] = 1e-3
        cfg["BATCH_SIZE"] = 16
        cfg["MAX_EPOCHS"] = 2  # Only 2 epochs for smoke test
        cfg["EARLY_STOP_PATIENCE"] = 1
        cfg["SEED"] = 42
        cfg["LOSS_ALPHA"] = 0.5

        device = torch.device("cpu")
        arch = "patchtst"
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Run training
        result = train_one_fold(fold, df_all, cfg, device, arch, str(ckpt_dir))

        # Check result structure
        assert "fold_id" in result
        assert "best_epoch" in result
        assert "best_val_rank_ic" in result
        assert "checkpoint_path" in result

        # Check values are reasonable
        assert result["fold_id"] == 0
        assert result["best_epoch"] >= 0
        assert isinstance(result["best_val_rank_ic"], float)

        # Check checkpoint was saved
        ckpt_path = result["checkpoint_path"]
        assert os.path.exists(ckpt_path)

        # Check checkpoint can be loaded
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "epoch" in ckpt
        assert "val_rank_ic" in ckpt
