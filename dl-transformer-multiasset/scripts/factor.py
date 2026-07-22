"""Factor inference and database emission with 12-column schema.

This module handles:
- predict_fold: Load checkpoint and run inference on test dates
- stitch_predictions: Concatenate and deduplicate predictions from multiple folds
- score_and_signal: Per-day rank and buy/sell/hold signal generation
- emit_factor_table: Attach metadata and enforce 12-column order
- main: Iterate folds, stitch, score, and write database.parquet
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from scripts.model import PatchTST
from scripts.train import pick_device
from scripts.utils import FACTOR_ID, FACTOR_NAME, ASSET_TYPE, DATA_VERSION


# ────────────────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────────────────


def predict_fold(
    feature_df: pd.DataFrame,
    ckpt_dir: Path,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Load checkpoint and run inference on test dates.

    Args:
        feature_df: Feature dataframe with columns [trade_date, asset_type, symbol, ...features...]
        ckpt_dir: Directory containing best_model.pt checkpoint and config
        device: Compute device (defaults to auto-select)

    Returns:
        DataFrame with [trade_date, asset_type, symbol, factor_value]

    Raises:
        FileNotFoundError: If checkpoint or config not found
        RuntimeError: If model loading fails
    """
    if device is None:
        device = pick_device()

    # Load checkpoint
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint (try both 'cfg' and 'model_config' keys)
    model_cfg = checkpoint.get("cfg") or checkpoint.get("model_config", {})
    lookback = model_cfg.get("LOOKBACK", 60)
    patch_len = model_cfg.get("PATCH_LEN", 16)
    stride = model_cfg.get("STRIDE", 8)
    d_model = model_cfg.get("D_MODEL", 128)
    n_heads = model_cfg.get("N_HEADS", 8)
    n_layers = model_cfg.get("N_LAYERS", 3)
    dropout = model_cfg.get("DROPOUT", 0.2)

    # Count input features (exclude metadata columns)
    # Match the columns excluded in train.py
    meta_cols = {"date", "symbol", "label", "ret_5d", "open", "high", "low", "close", "volume", "amount", "open_interest", "data_version"}
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    n_features = len(feature_cols)

    # Initialize model
    model = PatchTST(
        n_features=n_features,
        lookback=lookback,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Prepare input data
    # Group by symbol, sort by date, extract feature sequences
    predictions = []

    for symbol, group in feature_df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)

        # Extract feature matrix (exclude metadata)
        feature_matrix = group[feature_cols].values  # (T, n_features)

        # Skip if insufficient data
        if len(feature_matrix) < lookback:
            continue

        # Create rolling windows
        for i in range(lookback, len(feature_matrix) + 1):
            window = feature_matrix[i - lookback : i]  # (lookback, n_features)

            # Convert to tensor
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, lookback, n_features)

            # Run inference
            with torch.no_grad():
                pred = model(x)  # (1, 1)

            # Store prediction
            pred_value = pred.item()
            trade_date = group.loc[i - 1, "date"]  # Predict for current date

            predictions.append({
                "trade_date": trade_date,
                "asset_type": ASSET_TYPE,
                "symbol": symbol,
                "factor_value": pred_value,
            })

    return pd.DataFrame(predictions)


# ────────────────────────────────────────────────────────────────────────────
# Stitching
# ────────────────────────────────────────────────────────────────────────────


def stitch_predictions(all_folds: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate and deduplicate predictions from multiple folds.

    Args:
        all_folds: List of prediction DataFrames from different folds

    Returns:
        Deduplicated DataFrame sorted by trade_date, asset_type, symbol
    """
    if not all_folds:
        return pd.DataFrame(columns=["trade_date", "asset_type", "symbol", "factor_value"])

    # Concatenate all folds
    combined = pd.concat(all_folds, ignore_index=True)

    # Deduplicate: keep last occurrence (most recent fold)
    combined = combined.drop_duplicates(
        subset=["trade_date", "asset_type", "symbol"],
        keep="last",
    )

    # Sort by trade_date, asset_type, symbol
    combined = combined.sort_values(["trade_date", "asset_type", "symbol"]).reset_index(drop=True)

    return combined


# ────────────────────────────────────────────────────────────────────────────
# Scoring and Signaling
# ────────────────────────────────────────────────────────────────────────────


def score_and_signal(
    predictions: pd.DataFrame,
    buy_q: float = 0.1,
    sell_q: float = 0.1,
) -> pd.DataFrame:
    """Compute per-day rank and buy/sell/hold signals.

    Args:
        predictions: DataFrame with [trade_date, asset_type, symbol, factor_value]
        buy_q: Quantile threshold for buy signal (top buy_q get "buy")
        sell_q: Quantile threshold for sell signal (bottom sell_q get "sell")

    Returns:
        DataFrame with added columns [score, rank, signal]
    """
    df = predictions.copy()

    # Group by trade_date and compute rank
    def rank_and_signal_day(day_df: pd.DataFrame) -> pd.DataFrame:
        day_df = day_df.copy()

        # Rank: 1 (lowest factor_value) to N (highest), using first method for ties
        day_df["rank"] = day_df["factor_value"].rank(method="first").astype(int)

        # Score: normalized rank to [0, 1]
        max_rank = day_df["rank"].max()
        if max_rank > 1:
            day_df["score"] = (day_df["rank"] - 1) / (max_rank - 1)
        else:
            day_df["score"] = 0.5  # Single asset edge case

        # Signal: buy top buy_q, sell bottom sell_q, hold middle
        n = len(day_df)
        buy_threshold = int(n * (1 - buy_q))  # Top 10% means rank >= 90th percentile
        sell_threshold = int(n * sell_q)  # Bottom 10% means rank < 10th percentile

        def assign_signal(rank: int) -> str:
            if rank >= buy_threshold:
                return "buy"
            elif rank < sell_threshold:
                return "sell"
            else:
                return "hold"

        day_df["signal"] = day_df["rank"].apply(assign_signal)

        return day_df

    # Apply per-day ranking and signaling
    result_list = []
    for trade_date, day_df in df.groupby("trade_date"):
        scored_day = rank_and_signal_day(day_df)
        result_list.append(scored_day)

    result = pd.concat(result_list, ignore_index=True)

    return result


# ────────────────────────────────────────────────────────────────────────────
# Factor Table Emission
# ────────────────────────────────────────────────────────────────────────────


def emit_factor_table(
    scored: pd.DataFrame,
    update_time: pd.Timestamp,
) -> pd.DataFrame:
    """Attach metadata and enforce 12-column order.

    Args:
        scored: DataFrame with [trade_date, asset_type, symbol, factor_value, score, rank, signal]
        update_time: Timestamp when factor table was generated

    Returns:
        DataFrame with exact 12-column order:
        [trade_date, asset_type, symbol, factor_id, factor_name, factor_value,
         score, rank, signal, confidence, data_version, update_time]
    """
    df = scored.copy()

    # Add metadata columns
    df["factor_id"] = FACTOR_ID
    df["factor_name"] = FACTOR_NAME
    df["confidence"] = 1.0
    df["data_version"] = DATA_VERSION
    # Ensure datetime64[ns] dtype
    df["update_time"] = pd.to_datetime(update_time).as_unit('ns')

    # Enforce exact 12-column order
    result = df[[
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
    ]]

    return result


# ────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────


def main(
    feature_path: str,
    output_dir: str,
    fold_dirs: list[str],
    device_pref: str = "auto",
) -> None:
    """Iterate folds, stitch predictions, score, and write database.parquet.

    Args:
        feature_path: Path to features.parquet (with all dates)
        output_dir: Directory to write database.parquet
        fold_dirs: List of fold checkpoint directories (each contains best_model.pt)
        device_pref: Device preference ("auto", "cuda", "mps", "cpu")

    Outputs:
        Writes database.parquet to output_dir with 12-column schema
    """
    # Setup
    device = pick_device(device_pref)
    output_path = Path(output_dir) / "database.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load features
    print(f"Loading features from {feature_path}...")
    features_df = pd.read_parquet(feature_path)

    # Run inference on each fold
    all_predictions = []
    for fold_dir in fold_dirs:
        fold_path = Path(fold_dir)
        print(f"Running inference on fold: {fold_path.name}...")

        fold_predictions = predict_fold(features_df, fold_path, device)
        all_predictions.append(fold_predictions)

        print(f"  Generated {len(fold_predictions)} predictions")

    # Stitch predictions
    print("Stitching predictions...")
    stitched = stitch_predictions(all_predictions)
    print(f"Total unique predictions: {len(stitched)}")

    # Score and signal
    print("Computing scores and signals...")
    scored = score_and_signal(stitched)

    # Emit factor table
    update_time = pd.Timestamp.now()
    print(f"Generating factor table (update_time={update_time})...")
    factor_table = emit_factor_table(scored, update_time)

    # Write to parquet
    print(f"Writing to {output_path}...")
    factor_table.to_parquet(output_path, index=False)

    print(f"Factor table written: {len(factor_table)} rows, {len(factor_table.columns)} columns")
    print(f"Date range: {factor_table['trade_date'].min()} to {factor_table['trade_date'].max()}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python factor.py <feature_path> <output_dir> <fold_dir1> [fold_dir2 ...]")
        sys.exit(1)

    feature_path = sys.argv[1]
    output_dir = sys.argv[2]
    fold_dirs = sys.argv[3:]

    main(feature_path, output_dir, fold_dirs)
