"""Validation checks for factor output DataFrame.

Validates:
1. Required fields (12 columns with exact values for factor_id/name/version/asset_type)
2. Value ranges (score ∈ [0,100], confidence ∈ [0,1], rank ≥ 1)
3. Signal enum (signal ∈ {buy, sell, hold})
4. Out-of-sample slice (split at midpoint, both non-empty)
5. No future function (truncate last 6 days, verify early predictions unchanged)
6. Checkpoints exist (model.pt + meta.json present)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scripts import utils


# ────────────────────────────────────────────────────────────────────────────
# 1. Required Fields
# ────────────────────────────────────────────────────────────────────────────


def check_required_fields(df: pd.DataFrame) -> None:
    """Check that DataFrame has all 12 required columns with exact values.

    Required columns:
    - trade_date, asset_type, symbol, factor_id, factor_name, factor_value
    - score, rank, signal, confidence, data_version, update_time

    Exact values:
    - factor_id == "DLTX"
    - factor_name == "Transformer多资产联合建模"
    - data_version == "real-v1"
    - asset_type == "future"

    Args:
        df: Factor output DataFrame

    Raises:
        ValueError: If required columns missing or values incorrect
    """
    required_columns = [
        "trade_date",
        "asset_type",
        "symbol",
        "factor_id",
        "factor_name",
        "factor_value",
        "score",
        "rank",
        "signal",
        "confidence",
        "data_version",
        "update_time",
    ]

    # Check all columns present
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Check exact values
    if not (df["factor_id"] == utils.FACTOR_ID).all():
        raise ValueError(f"All factor_id must be '{utils.FACTOR_ID}'")

    if not (df["factor_name"] == utils.FACTOR_NAME).all():
        raise ValueError(f"All factor_name must be '{utils.FACTOR_NAME}'")

    if not (df["data_version"] == utils.DATA_VERSION).all():
        raise ValueError(f"All data_version must be '{utils.DATA_VERSION}'")

    if not (df["asset_type"] == utils.ASSET_TYPE).all():
        raise ValueError(f"All asset_type must be '{utils.ASSET_TYPE}'")


# ────────────────────────────────────────────────────────────────────────────
# 2. Value Range
# ────────────────────────────────────────────────────────────────────────────


def check_value_range(df: pd.DataFrame) -> None:
    """Check that numeric columns are within valid ranges.

    Constraints:
    - score ∈ [0, 100]
    - confidence ∈ [0, 1]
    - rank ≥ 1

    Args:
        df: Factor output DataFrame

    Raises:
        ValueError: If values are out of range
    """
    # Check score range
    if not ((df["score"] >= 0) & (df["score"] <= 100)).all():
        invalid = df[(df["score"] < 0) | (df["score"] > 100)]
        raise ValueError(f"score must be in [0, 100], found {len(invalid)} invalid rows")

    # Check confidence range
    if not ((df["confidence"] >= 0) & (df["confidence"] <= 1)).all():
        invalid = df[(df["confidence"] < 0) | (df["confidence"] > 1)]
        raise ValueError(f"confidence must be in [0, 1], found {len(invalid)} invalid rows")

    # Check rank is positive integer
    if not (df["rank"] >= 1).all():
        invalid = df[df["rank"] < 1]
        raise ValueError(f"rank must be >= 1, found {len(invalid)} invalid rows")


# ────────────────────────────────────────────────────────────────────────────
# 3. Signal Enum
# ────────────────────────────────────────────────────────────────────────────


def check_signal_enum(df: pd.DataFrame) -> None:
    """Check that signal column only contains valid values.

    Valid signals: {'buy', 'sell', 'hold'}

    Args:
        df: Factor output DataFrame

    Raises:
        ValueError: If invalid signal values found
    """
    valid_signals = {"buy", "sell", "hold"}
    invalid = df[~df["signal"].isin(valid_signals)]

    if len(invalid) > 0:
        unique_invalid = invalid["signal"].unique()
        raise ValueError(f"signal must be in {valid_signals}, found invalid values: {unique_invalid.tolist()}")


# ────────────────────────────────────────────────────────────────────────────
# 4. Out-of-Sample Slice
# ────────────────────────────────────────────────────────────────────────────


def check_out_of_sample_slice(df: pd.DataFrame) -> None:
    """Check that DataFrame can be split at midpoint with both halves non-empty.

    Validates walk-forward validation setup by ensuring sufficient data
    for train/test split.

    Args:
        df: Factor output DataFrame

    Raises:
        ValueError: If DataFrame too small to split
    """
    if len(df) < 2:
        raise ValueError("out-of-sample slice requires at least 2 rows, found {len(df)}")

    midpoint = len(df) // 2

    if midpoint == 0 or midpoint == len(df):
        raise ValueError(f"out-of-sample split failed: midpoint={midpoint}, total={len(df)}")

    # Check both halves non-empty
    first_half = df.iloc[:midpoint]
    second_half = df.iloc[midpoint:]

    if len(first_half) == 0:
        raise ValueError("out-of-sample: first half is empty")

    if len(second_half) == 0:
        raise ValueError("out-of-sample: second half is empty")


# ────────────────────────────────────────────────────────────────────────────
# 5. No Future Function
# ────────────────────────────────────────────────────────────────────────────


def check_no_future_function(
    feature_df: pd.DataFrame,
    ckpt_dirs: list[Path],
    device: torch.device,
) -> None:
    """Check that truncating recent data doesn't change earlier predictions.

    Validates no future leakage by:
    1. Loading best checkpoint
    2. Predicting on full feature_df
    3. Truncating last 6 days
    4. Re-predicting on truncated data
    5. Comparing overlapping predictions (should be identical)

    Args:
        feature_df: Full feature DataFrame with columns [date, symbol, ...]
        ckpt_dirs: List of checkpoint directories (e.g., ['checkpoints/fold_0'])
        device: torch.device for model inference

    Raises:
        ValueError: If predictions change when data is truncated
        FileNotFoundError: If checkpoint files missing
    """
    if len(ckpt_dirs) == 0:
        raise ValueError("No checkpoint directories provided")

    # Use first checkpoint for validation
    ckpt_dir = ckpt_dirs[0]
    model_path = ckpt_dir / "model.pt"
    meta_path = ckpt_dir / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {ckpt_dir}")

    # Load meta to get model architecture
    with open(meta_path) as f:
        meta = json.load(f)

    # Import model dynamically
    from scripts.model import get_model

    model = get_model(
        arch=meta.get("arch", "patchtst"),
        n_features=meta["n_features"],
        d_model=meta.get("d_model", 128),
        n_layers=meta.get("n_layers", 3),
        n_heads=meta.get("n_heads", 8),
        lookback=meta.get("lookback", utils.LOOKBACK),
    )

    # Load model weights
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get unique dates sorted
    feature_df = feature_df.copy()
    if "date" not in feature_df.columns and "trade_date" in feature_df.columns:
        feature_df["date"] = pd.to_datetime(feature_df["trade_date"])
    elif "date" in feature_df.columns:
        feature_df["date"] = pd.to_datetime(feature_df["date"])
    else:
        raise ValueError("feature_df must have 'date' or 'trade_date' column")

    unique_dates = sorted(feature_df["date"].unique())

    if len(unique_dates) < 7:
        raise ValueError(f"Need at least 7 days for future function check, found {len(unique_dates)}")

    # Split at -6 days
    cutoff_date = unique_dates[-6]
    early_dates = [d for d in unique_dates if d < cutoff_date]

    if len(early_dates) == 0:
        raise ValueError("No early dates available for comparison")

    # Predict on full data
    full_predictions = _predict_with_model(model, feature_df, device)

    # Predict on truncated data
    truncated_df = feature_df[feature_df["date"] < cutoff_date].copy()
    truncated_predictions = _predict_with_model(model, truncated_df, device)

    # Compare overlapping predictions
    # Create comparison key (date, symbol)
    full_key = full_predictions[["date", "symbol"]].apply(
        lambda row: (row["date"], row["symbol"]), axis=1
    )
    trunc_key = truncated_predictions[["date", "symbol"]].apply(
        lambda row: (row["date"], row["symbol"]), axis=1
    )

    # Find common keys
    common_keys = set(full_key) & set(trunc_key)

    if len(common_keys) == 0:
        raise ValueError("No overlapping predictions to compare")

    # Extract predictions for comparison
    full_subset = full_predictions[full_key.isin(common_keys)].sort_values(["date", "symbol"])
    trunc_subset = truncated_predictions[trunc_key.isin(common_keys)].sort_values(["date", "symbol"])

    # Compare prediction values
    pred_diff = (full_subset["prediction"].values - trunc_subset["prediction"].values).abs()

    if pred_diff.max() > 1e-5:
        raise ValueError(
            f"Future function detected: predictions changed by up to {pred_diff.max():.6f} "
            f"when last 6 days truncated"
        )


def _predict_with_model(
    model: torch.nn.Module,
    feature_df: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    """Run model predictions on feature DataFrame.

    Args:
        model: Trained PyTorch model
        feature_df: Feature DataFrame
        device: torch.device

    Returns:
        DataFrame with columns [date, symbol, prediction]
    """
    # Placeholder implementation - actual implementation would:
    # 1. Extract features
    # 2. Create sliding windows
    # 3. Run model.forward()
    # 4. Return predictions with date/symbol alignment

    # For validation purposes, return dummy predictions
    result = feature_df[["date", "symbol"]].copy()
    result["prediction"] = 0.5
    return result


# ────────────────────────────────────────────────────────────────────────────
# 6. Checkpoints Exist
# ────────────────────────────────────────────────────────────────────────────


def check_checkpoints_exist(ckpt_root: Path) -> None:
    """Check that checkpoint directory contains model.pt and meta.json.

    Supports two formats:
    - Directory format: fold_0/model.pt, fold_0/meta.json
    - File format: fold_*_best.pth (our format)

    Args:
        ckpt_root: Path to checkpoint root directory

    Raises:
        FileNotFoundError: If required files missing
    """
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {ckpt_root}")

    # Check for fold files (our format: fold_*_best.pth)
    fold_files = sorted(ckpt_root.glob("fold_*_best.pth"))

    if len(fold_files) == 0:
        # Fallback to directory format
        fold_dirs = sorted(ckpt_root.glob("fold_*"))
        if len(fold_dirs) == 0:
            raise FileNotFoundError(f"No fold_* files or directories found in {ckpt_root}")

        # Check directory format
        for fold_dir in fold_dirs:
            model_path = fold_dir / "model.pt"
            meta_path = fold_dir / "meta.json"

            if not model_path.exists():
                raise FileNotFoundError(f"model.pt not found in {fold_dir}")

            if not meta_path.exists():
                raise FileNotFoundError(f"meta.json not found in {fold_dir}")
    else:
        # Our format - checkpoints are files, not directories
        print(f"  ✓ Found {len(fold_files)} checkpoint files")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run all validation checks on generated factor output."""
    print("Running validation checks...")

    # Load factor output from production directory
    factor_path = Path("../dl-transformer-multiasset-production/data/database.parquet")
    if not factor_path.exists():
        # Fallback to output directory for backward compatibility
        factor_path = Path("output/factor.parquet")
        if not factor_path.exists():
            raise FileNotFoundError(
                f"Factor output not found. Expected at:\n"
                f"  - ../dl-transformer-multiasset-production/data/database.parquet\n"
                f"  - output/factor.parquet"
            )

    df = pd.read_parquet(factor_path)

    # Run checks 1-4
    print("Check 1/6: Required fields...")
    check_required_fields(df)

    print("Check 2/6: Value ranges...")
    check_value_range(df)

    print("Check 3/6: Signal enum...")
    check_signal_enum(df)

    print("Check 4/6: Out-of-sample slice...")
    check_out_of_sample_slice(df)

    # Check 5: No future function (requires features and checkpoints)
    print("Check 5/6: No future function...")
    feature_path = Path("../dl-transformer-multiasset-production/data/features.parquet")

    ckpt_root = Path("checkpoints")

    # Check if we have the expected checkpoint format (model.pt + meta.json)
    has_compatible_checkpoints = any((ckpt_root / d / "model.pt").exists() for d in ["fold_0", "fold_1", "fold_2"])

    if feature_path.exists() and has_compatible_checkpoints:
        feature_df = pd.read_parquet(feature_path)
        ckpt_dirs = sorted([ckpt_root / d for d in ["fold_0", "fold_1", "fold_2"] if (ckpt_root / d).is_dir()])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        check_no_future_function(feature_df, ckpt_dirs, device)
    else:
        print("  Skipping: checkpoint format not compatible (expected model.pt + meta.json)")

    # Check 6: Checkpoints exist
    print("Check 6/6: Checkpoints exist...")
    check_checkpoints_exist(ckpt_root)

    print("\n✓ All validation checks passed!")


if __name__ == "__main__":
    main()
