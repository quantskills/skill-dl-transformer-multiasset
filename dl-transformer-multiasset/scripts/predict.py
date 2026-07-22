"""Feature building and model inference for out-of-sample prediction pipeline.

This module handles:
- Loading raw data for prediction period with 80-day lookback buffer
- Computing engineered features with cross-sectional normalization
- Running model inference on fold_2 checkpoint
- Scoring and signaling predictions
- Writing predict_features.parquet and predict.parquet to production directory
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from scripts.features import (
    list_commodity_symbols,
    load_raw_panel,
    add_engineered_features,
    xsec_zscore,
    add_label,
)
from scripts.utils import _get_env, _date_to_yyyymmdd
from scripts import factor
from scripts.train import pick_device
from scripts.backtest import (
    load_real_dominant_and_daily,
    build_forward_returns,
    build_tradeable_forward_returns,
    calculate_metrics,
    count_rollovers,
    calculate_ic_by_horizon,
)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

_LOOKBACK_BUFFER_DAYS = 80
PRODUCTION_DATA_DIR = Path(__file__).parent.parent.parent / "dl-transformer-multiasset-production" / "data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
DATA_VERSION_PREDICT = "predict-v1"


# ────────────────────────────────────────────────────────────────────────────
# Date Utilities
# ────────────────────────────────────────────────────────────────────────────


def _subtract_days(date_str: str, days: int) -> str:
    """Subtract days from a YYYYMMDD date string.

    Args:
        date_str: Date string in YYYYMMDD format
        days: Number of days to subtract

    Returns:
        Date string in YYYYMMDD format after subtracting days
    """
    dt = datetime.strptime(date_str, "%Y%m%d")
    dt_new = dt - timedelta(days=days)
    return dt_new.strftime("%Y%m%d")


# ────────────────────────────────────────────────────────────────────────────
# Feature Building
# ────────────────────────────────────────────────────────────────────────────


def build_predict_features(
    predict_start: str | None = None,
    predict_end: str | None = None,
) -> pd.DataFrame:
    """Build feature frame for prediction period with lookback context.

    Orchestrates the full pipeline:
    1. Load raw panel data with 80-day lookback buffer before predict_start
    2. Add engineered features
    3. Drop rows with NaN in mom_60 (insufficient lookback)
    4. Apply cross-sectional z-score normalization
    5. Add forward return labels (gracefully handle failures)
    6. Write predict_features.parquet to production directory

    Args:
        predict_start: Start date in YYYYMMDD format (or from PANDA_DATA_PREDICT_START env var)
        predict_end: End date in YYYYMMDD format (or from PANDA_DATA_PREDICT_END env var)

    Returns:
        DataFrame with complete feature set, labels, and data_version

    Raises:
        RuntimeError: If environment variables PANDA_DATA_PREDICT_START or
                     PANDA_DATA_PREDICT_END are not set and not provided as arguments
    """
    # Get dates from arguments or environment
    if predict_start is None:
        predict_start = _get_env("PANDA_DATA_PREDICT_START")
    if predict_end is None:
        predict_end = _get_env("PANDA_DATA_PREDICT_END")

    # Normalize date format
    predict_start = _date_to_yyyymmdd(predict_start)
    predict_end = _date_to_yyyymmdd(predict_end)

    # Calculate lookback buffer start date
    buffer_start = _subtract_days(predict_start, _LOOKBACK_BUFFER_DAYS)

    # Load symbols
    symbols = list_commodity_symbols()

    # Load raw panel with lookback buffer
    print(f"Loading raw panel from {buffer_start} to {predict_end}...")
    raw = load_raw_panel(symbols, buffer_start, predict_end)

    # Add engineered features
    print(f"Adding engineered features...")
    feat = add_engineered_features(raw)

    # Drop rows with NaN in mom_60 (insufficient lookback)
    print(f"Dropping rows with insufficient lookback (mom_60 NaN)...")
    feat = feat.dropna(subset=["mom_60"]).reset_index(drop=True)

    # Apply cross-sectional normalization
    print(f"Applying cross-sectional z-score normalization...")
    feat = xsec_zscore(feat, exclude=("date", "symbol"))

    # Add forward return labels (gracefully handle failures)
    print(f"Adding forward return labels...")
    try:
        feat = add_label(feat, horizon=5)
    except Exception as e:
        print(f"Warning: Failed to add labels: {e}")
        feat["ret_5d"] = float('nan')
        feat["label"] = float('nan')

    # Add data version
    feat["data_version"] = DATA_VERSION_PREDICT

    # Write to production directory
    print(f"Writing to production directory...")
    PRODUCTION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PRODUCTION_DATA_DIR / "predict_features.parquet"
    feat.to_parquet(output_path, index=False)
    print(f"Wrote {len(feat)} rows to {output_path}")

    return feat


# ────────────────────────────────────────────────────────────────────────────
# Model Inference
# ────────────────────────────────────────────────────────────────────────────


def run_prediction(
    predict_start: str | None = None,
    predict_end: str | None = None,
) -> pd.DataFrame:
    """Load fold_2 checkpoint, run inference, score, and emit factor table.

    Orchestrates the full prediction pipeline:
    1. Load predict_features.parquet from production directory
    2. Filter to date range [predict_start, predict_end]
    3. Load fold_2 checkpoint and run inference
    4. Score and signal per-day rankings
    5. Emit 12-column factor table with data_version="predict-v1"
    6. Write predict.parquet to production directory

    Args:
        predict_start: Start date in YYYYMMDD format (or from env var)
        predict_end: End date in YYYYMMDD format (or from env var)

    Returns:
        DataFrame with 12 columns:
        [trade_date, asset_type, symbol, factor_id, factor_name, factor_value,
         score, rank, signal, confidence, data_version, update_time]

    Raises:
        FileNotFoundError: If predict_features.parquet or fold_2 checkpoint missing
        RuntimeError: If model loading fails
    """
    # Get dates from arguments or environment
    if predict_start is None:
        predict_start = _get_env("PANDA_DATA_PREDICT_START")
    if predict_end is None:
        predict_end = _get_env("PANDA_DATA_PREDICT_END")

    # Normalize date format
    predict_start = _date_to_yyyymmdd(predict_start)
    predict_end = _date_to_yyyymmdd(predict_end)

    # Load predict_features.parquet
    print(f"Loading predict_features.parquet from {PRODUCTION_DATA_DIR}...")
    features_path = PRODUCTION_DATA_DIR / "predict_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features_df = pd.read_parquet(features_path)
    print(f"Loaded {len(features_df)} feature rows")

    # Filter to date range
    print(f"Filtering to date range {predict_start} to {predict_end}...")
    features_df = features_df[
        (features_df["date"] >= predict_start) &
        (features_df["date"] <= predict_end)
    ].reset_index(drop=True)
    print(f"After filtering: {len(features_df)} rows")

    # Setup device
    device = pick_device()
    print(f"Using device: {device}")

    # Create a temporary directory for fold structure
    # factor.predict_fold expects a directory with best_model.pt
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fold_dir = Path(tmpdir) / "fold_2"
        tmp_fold_dir.mkdir(parents=True, exist_ok=True)

        # Check if fold_2 checkpoint exists
        fold_2_checkpoint = CHECKPOINT_DIR / "fold_2_best.pth"
        if not fold_2_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {fold_2_checkpoint}")

        # Copy checkpoint to temporary location with expected name
        import shutil
        shutil.copy(fold_2_checkpoint, tmp_fold_dir / "best_model.pt")
        print(f"Copied checkpoint to {tmp_fold_dir}")

        # Run inference
        print(f"Running inference on fold_2...")
        predictions = factor.predict_fold(features_df, tmp_fold_dir, device)
        print(f"Generated {len(predictions)} predictions")

    # Score and signal
    print(f"Computing scores and signals...")
    scored = factor.score_and_signal(predictions)

    # Emit factor table
    update_time = pd.Timestamp.now()
    print(f"Generating factor table (update_time={update_time})...")
    factor_table = factor.emit_factor_table(scored, update_time)

    # Ensure data_version is "predict-v1"
    factor_table["data_version"] = DATA_VERSION_PREDICT

    # Write to parquet
    print(f"Writing to production directory...")
    PRODUCTION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PRODUCTION_DATA_DIR / "predict.parquet"
    factor_table.to_parquet(output_path, index=False)
    print(f"Wrote {len(factor_table)} rows to {output_path}")

    return factor_table


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────


def evaluate_predictions(
    predict_start: str | None = None,
    predict_end: str | None = None,
) -> dict | None:
    """Evaluate prediction performance with backtest on forward returns.

    Orchestrates the evaluation pipeline:
    1. Load predict.parquet from production directory
    2. Return None if predict.parquet doesn't exist
    3. Load forward returns data using backtest functions
    4. Compute two metric buckets: "research" (no lag, no cost) and "tradeable" (1-day lag, 5bps)
    5. Write predict_report.json to production directory

    Args:
        predict_start: Start date in YYYYMMDD format (or from env var)
        predict_end: End date in YYYYMMDD format (or from env var)

    Returns:
        Dictionary with keys "research", "tradeable", "predict_period", etc., or None if:
        - predict.parquet doesn't exist
        - No forward returns available (prediction period is in the future)

    Raises:
        None - gracefully handles exceptions by printing them
    """
    try:
        # Get dates from arguments or environment
        if predict_start is None:
            predict_start = _get_env("PANDA_DATA_PREDICT_START")
        if predict_end is None:
            predict_end = _get_env("PANDA_DATA_PREDICT_END")

        # Normalize date format
        predict_start = _date_to_yyyymmdd(predict_start)
        predict_end = _date_to_yyyymmdd(predict_end)

        # Load predict.parquet
        predictions_path = PRODUCTION_DATA_DIR / "predict.parquet"
        if not predictions_path.exists():
            print(f"Info: predict.parquet not found at {predictions_path}")
            return None

        print(f"Loading predict.parquet...")
        predictions_df = pd.read_parquet(predictions_path)
        print(f"Loaded {len(predictions_df)} prediction rows")

        # Extract symbols and date range from predictions
        symbols = sorted(predictions_df["symbol"].unique().tolist())
        prediction_start_date = predictions_df["trade_date"].min()
        prediction_end_date = predictions_df["trade_date"].max()

        print(f"Prediction period: {prediction_start_date} to {prediction_end_date}")
        print(f"Symbols: {symbols}")

        # Extract underlying commodity codes from full contract symbols
        # e.g., "AG_DOMINANT.SHF" -> "AG"
        underlying_symbols = sorted(set([s.split("_")[0] for s in symbols]))
        print(f"Underlying commodities: {underlying_symbols}")

        # Load forward returns using backtest functions
        print(f"Loading forward returns data...")
        try:
            dominant, daily = load_real_dominant_and_daily(underlying_symbols, prediction_start_date, prediction_end_date)
        except Exception as e:
            print(f"Warning: Failed to load forward returns data: {e}")
            return None

        # Build forward returns (research bucket: no lag, no cost)
        try:
            forward_research = build_forward_returns(dominant, daily)
            print(f"Research forward returns: {len(forward_research)} rows")
            if len(forward_research) > 0:
                print(f"  Sample columns: {list(forward_research.columns)}")
                print(f"  Sample data (first 3 rows):\n{forward_research.head(3)}")
        except Exception as e:
            print(f"Warning: Failed to build research forward returns: {e}")
            forward_research = pd.DataFrame()

        # Build tradeable forward returns (1-day lag, 5bps cost)
        try:
            forward_tradeable = build_tradeable_forward_returns(
                dominant, daily,
                data_lag=1,
                roll_cost_bps=5.0,
            )
            print(f"Tradeable forward returns: {len(forward_tradeable)} rows")
        except Exception as e:
            print(f"Warning: Failed to build tradeable forward returns: {e}")
            forward_tradeable = pd.DataFrame()

        # If no forward returns available, return None
        if forward_research.empty and forward_tradeable.empty:
            print(f"Warning: No forward returns available for evaluation")
            return None

        # Count rollovers
        try:
            rollover_count = count_rollovers(dominant)
            print(f"Rollover count: {rollover_count}")
        except Exception as e:
            print(f"Warning: Failed to count rollovers: {e}")
            rollover_count = 0

        # Rename trade_date to trade_date for consistency with calculate_metrics
        # First, we need to ensure predictions_df has the right column names
        predictions_for_backtest = predictions_df.copy()

        # Convert symbol format from 'AG_DOMINANT.SHF' to 'AG' and trade_date to match format
        predictions_for_backtest["symbol"] = predictions_for_backtest["symbol"].str.split("_").str[0]
        # Convert trade_date from YYYYMMDD to YYYY-MM-DD format
        predictions_for_backtest["trade_date"] = pd.to_datetime(
            predictions_for_backtest["trade_date"], format="%Y%m%d"
        ).dt.strftime("%Y-%m-%d")

        # Compute metrics for research bucket
        research_metrics = None
        if not forward_research.empty:
            try:
                print(f"Computing research metrics (no lag, no cost)...")
                print(f"  Predictions columns: {list(predictions_for_backtest.columns)}")
                print(f"  Predictions (first 2):\n{predictions_for_backtest[['trade_date', 'symbol']].head(2)}")
                print(f"  Forward returns columns: {list(forward_research.columns)}")
                print(f"  Forward returns (first 2):\n{forward_research[['trade_date', 'symbol']].head(2)}")
                research_metrics = calculate_metrics(
                    predictions_for_backtest,
                    forward_research,
                    rollover_count=rollover_count,
                )
                # Also compute IC by horizon
                try:
                    research_metrics["IC_by_horizon"] = calculate_ic_by_horizon(
                        predictions_for_backtest,
                        dominant,
                        daily,
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute IC by horizon: {e}")
                    research_metrics["IC_by_horizon"] = {}

                print(f"Research metrics computed")
            except Exception as e:
                print(f"Warning: Failed to compute research metrics: {e}")
                research_metrics = None

        # Compute metrics for tradeable bucket
        tradeable_metrics = None
        if not forward_tradeable.empty:
            try:
                print(f"Computing tradeable metrics (1-day lag, 5bps)...")
                tradeable_metrics = calculate_metrics(
                    predictions_for_backtest,
                    forward_tradeable,
                    rollover_count=rollover_count,
                )
                tradeable_metrics["data_lag"] = 1
                tradeable_metrics["roll_cost_bps"] = 5.0

                # Also compute IC by horizon
                try:
                    tradeable_metrics["IC_by_horizon"] = calculate_ic_by_horizon(
                        predictions_for_backtest,
                        dominant,
                        daily,
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute IC by horizon: {e}")
                    tradeable_metrics["IC_by_horizon"] = {}

                print(f"Tradeable metrics computed")
            except Exception as e:
                print(f"Warning: Failed to compute tradeable metrics: {e}")
                tradeable_metrics = None

        # Build result dictionary
        result = {
            "predict_period": f"{prediction_start_date} to {prediction_end_date}",
            "n_symbols": len(symbols),
            "n_trading_days": predictions_df["trade_date"].nunique(),
            "n_predictions": len(predictions_df),
            "research": research_metrics,
            "tradeable": tradeable_metrics,
            "generated_at": datetime.now().isoformat(),
        }

        # Write predict_report.json
        print(f"Writing predict_report.json...")
        report_path = PRODUCTION_DATA_DIR / "predict_report.json"
        PRODUCTION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote report to {report_path}")

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point for out-of-sample prediction pipeline.

    Usage:
        python -m scripts.predict --step features|predict|evaluate|all

    Environment variables:
        PANDA_DATA_PREDICT_START: Prediction start date (YYYYMMDD)
        PANDA_DATA_PREDICT_END: Prediction end date (YYYYMMDD)
        PANDA_DATA_USERNAME: Panda data API username
        PANDA_DATA_PASSWORD: Panda data API password
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Out-of-Sample Prediction Pipeline"
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["features", "predict", "evaluate", "all"],
        help="Pipeline step to run: features, predict, evaluate, or all",
    )
    args = parser.parse_args()

    if args.step == "features" or args.step == "all":
        print("=" * 50)
        print("Step 1: Building prediction features")
        print("=" * 50)
        build_predict_features()

    if args.step == "predict" or args.step == "all":
        print("\n" + "=" * 50)
        print("Step 2: Running model inference")
        print("=" * 50)
        run_prediction()

    if args.step == "evaluate" or args.step == "all":
        print("\n" + "=" * 50)
        print("Step 3: Evaluating predictions")
        print("=" * 50)
        result = evaluate_predictions()
        if result is None:
            print("Evaluation skipped (no forward returns available).")


def _login_panda_data():
    """Initialize panda_data client with credentials from environment."""
    import os
    import panda_data

    username = os.environ.get("PANDA_DATA_USERNAME")
    password = os.environ.get("PANDA_DATA_PASSWORD")
    if username and password:
        try:
            panda_data.init_token(username=username, password=password)
            print("[INFO] Panda Data login successful")
        except Exception as e:
            print(f"[WARNING] Panda Data login failed: {e}")
    else:
        print("[INFO] PANDA_DATA_USERNAME/PASSWORD not set, skipping login (evaluation will not be available)")


if __name__ == "__main__":
    _login_panda_data()
    main()
