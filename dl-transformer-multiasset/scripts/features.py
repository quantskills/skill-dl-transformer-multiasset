"""Feature engineering and data loading for Transformer Multi-Asset skill.

This module handles:
- Data loading from panda_data API (get_future_detail, get_future_daily_post)
- Symbol normalization (contract → underlying)
- Feature engineering (momentum, reversal, volatility, volume, OI, amount features)
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import panda_data

from scripts.utils import (
    _batched,
    _call_with_retry,
    _date_to_iso,
    _date_to_yyyymmdd,
    _get_env,
)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
_BATCH = 5
_BATCH_SLEEP = 2.0

# ────────────────────────────────────────────────────────────────────────────
# Symbol Listing
# ────────────────────────────────────────────────────────────────────────────


def list_commodity_symbols() -> list[str]:
    """List all commodity futures underlying symbols from panda_data.

    Calls panda_data.get_future_detail, filters for product=='commodity',
    and returns sorted list of underlying symbols.

    Returns:
        Sorted list of commodity underlying symbols (e.g., ['AG', 'AL', 'AU', ...])

    Raises:
        RuntimeError: If PANDA_DATA_USERNAME or PANDA_DATA_PASSWORD not set
        Exception: If panda_data API call fails
    """
    # Initialize token with credentials from environment
    username = _get_env("PANDA_DATA_USERNAME")
    password = _get_env("PANDA_DATA_PASSWORD")
    panda_data.init_token(username=username, password=password)

    # Fetch futures detail (only actively trading contracts to avoid package limit)
    df = _call_with_retry(panda_data.get_future_detail, is_trading=1)

    # Filter for commodities (case-insensitive)
    df_commodity = df[df["product"].str.lower() == "commodity"].copy()

    # Get unique underlying symbols and sort
    symbols = sorted(df_commodity["underlying_symbol"].unique().tolist())

    return symbols


# ────────────────────────────────────────────────────────────────────────────
# Symbol Normalization
# ────────────────────────────────────────────────────────────────────────────


def _normalize_symbol_col(df: pd.DataFrame, src_col: str = "symbol") -> pd.DataFrame:
    """Normalize contract symbols to underlying symbols.

    Maps contract codes (e.g., 'RB2410') to underlying symbols (e.g., 'RB')
    by stripping trailing digits.

    Args:
        df: DataFrame with contract symbols
        src_col: Column name containing contract symbols

    Returns:
        DataFrame with normalized symbol column
    """
    df = df.copy()
    # Strip trailing digits to get underlying symbol
    # Pattern: match letters at start, ignore trailing digits
    df[src_col] = df[src_col].str.replace(r"\d+$", "", regex=True)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Data Loading
# ────────────────────────────────────────────────────────────────────────────


def _fetch_daily_post_batch(
    symbols: list[str], start: str, end: str
) -> pd.DataFrame:
    """Fetch daily post-market data for a batch of symbols.

    Wrapper around panda_data.get_future_daily_post with retry logic.

    Args:
        symbols: List of underlying symbols (e.g., ['RB', 'AU'])
        start: Start date in YYYYMMDD or YYYY-MM-DD format
        end: End date in YYYYMMDD or YYYY-MM-DD format

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume, amount, open_interest
    """
    # Convert dates to ISO format for API
    start_iso = _date_to_iso(start)
    end_iso = _date_to_iso(end)

    # Fetch data with retry
    df = _call_with_retry(
        panda_data.get_future_daily_post,
        underlying_symbol=symbols,
        start_date=start_iso,
        end_date=end_iso,
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "open_interest",
            ]
        )

    # Normalize symbol column if it contains contract codes
    df = _normalize_symbol_col(df, src_col="symbol")

    # Convert date to YYYYMMDD string format
    if "date" in df.columns:
        # Handle various date formats from API
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

    # Select and order required columns
    required_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "open_interest",
    ]

    # Ensure all columns exist and are in correct order
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df = df[required_cols].copy()

    # Convert numeric columns to float
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "open_interest"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_raw_panel(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Load raw OHLCV panel data for multiple symbols.

    Fetches daily post-market data from panda_data in batches (5 symbols per batch,
    2s sleep between batches). Returns long-form DataFrame with one row per
    symbol-date observation.

    For contracts with multiple observations per day, keeps only the dominant
    contract (highest volume).

    Args:
        symbols: List of underlying symbols (e.g., ['RB', 'AU', 'CU'])
        start: Start date in YYYYMMDD or YYYY-MM-DD format
        end: End date in YYYYMMDD or YYYY-MM-DD format

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume, amount, open_interest
        Sorted by date and symbol. Date is YYYYMMDD string, all prices/volumes are float.

    Raises:
        RuntimeError: If PANDA_DATA_USERNAME or PANDA_DATA_PASSWORD not set
        Exception: If panda_data API call fails after retries
    """
    # Initialize token
    username = _get_env("PANDA_DATA_USERNAME")
    password = _get_env("PANDA_DATA_PASSWORD")
    panda_data.init_token(username=username, password=password)

    # Fetch data in batches
    dfs = []
    for i, batch in enumerate(_batched(symbols, _BATCH)):
        if i > 0:
            time.sleep(_BATCH_SLEEP)

        df_batch = _fetch_daily_post_batch(batch, start, end)
        if not df_batch.empty:
            dfs.append(df_batch)

    if not dfs:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "open_interest",
            ]
        )

    # Concatenate all batches
    df = pd.concat(dfs, ignore_index=True)

    # Keep dominant contract per day (highest volume)
    df = (
        df.sort_values("volume", ascending=False)
        .drop_duplicates(subset=["date", "symbol"], keep="first")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )

    return df


# ────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ────────────────────────────────────────────────────────────────────────────

# Feature engineering windows
MOM_WINDOWS = [5, 10, 20, 60]
REV_WINDOWS = [5, 20]
VOL_WINDOWS = [5, 20, 60]
SHORT_WINDOWS = [5, 20]
STAT_WINDOW = 20


def _group_rolling(df: pd.DataFrame, col: str, window: int, op: str) -> pd.Series:
    """Apply rolling operation grouped by symbol.

    Args:
        df: DataFrame with 'symbol' column
        col: Column name to apply rolling operation on
        window: Rolling window size
        op: Operation - 'mean', 'std', or 'sum'

    Returns:
        Series with rolling operation applied per symbol group
    """
    grp = df.groupby("symbol")[col]
    if op == "mean":
        return grp.transform(lambda s: s.rolling(window, min_periods=window).mean())
    if op == "std":
        return grp.transform(lambda s: s.rolling(window, min_periods=window).std())
    if op == "sum":
        return grp.transform(lambda s: s.rolling(window, min_periods=window).sum())
    raise ValueError(f"Unknown operation: {op}")


def add_engineered_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to raw panel data.

    Takes 9-column raw panel (date, symbol, open, high, low, close, volume, amount,
    open_interest) and returns same rows with ~75 feature columns added:
    - Base (7): open, high, low, close, volume, amount, open_interest
    - Derived (17): momentum, reversal, volatility, volume ratios, OI change, amount features
    - Rolling stats (51): for each derived feature, add _mean20, _std20, _diff1

    All rolling operations use groupby("symbol") to prevent lookahead across symbols.
    All numeric outputs are float32.

    Args:
        raw: DataFrame with columns [date, symbol, open, high, low, close, volume, amount, open_interest]

    Returns:
        DataFrame with all original columns plus ~75 engineered features
    """
    df = raw.sort_values(["symbol", "date"]).reset_index(drop=True).copy()

    # log return (used for vol_N)
    log_return = np.log(df["close"]).groupby(df["symbol"]).diff()

    derived: dict[str, pd.Series] = {}

    # Momentum: (close / close_shifted_N) - 1
    for n in MOM_WINDOWS:
        shifted = df.groupby("symbol")["close"].shift(n)
        derived[f"mom_{n}"] = df["close"] / shifted - 1

    # Reversal: negative z-score of price vs rolling mean
    for n in REV_WINDOWS:
        ma = _group_rolling(df, "close", n, "mean")
        sd = _group_rolling(df, "close", n, "std")
        derived[f"rev_{n}"] = -(df["close"] - ma) / sd

    # Volatility: rolling std of log returns
    tmp = df.copy()
    tmp["_lr"] = log_return
    for n in VOL_WINDOWS:
        derived[f"vol_{n}"] = _group_rolling(tmp, "_lr", n, "std")

    # Volume ratio, OI change, amount ratio, vwap deviation
    for n in SHORT_WINDOWS:
        # Volume ratio
        vol_ma = _group_rolling(df, "volume", n, "mean")
        derived[f"vol_ratio_{n}"] = df["volume"] / vol_ma

        # OI change
        oi_shift = df.groupby("symbol")["open_interest"].shift(n)
        derived[f"oi_change_{n}"] = df["open_interest"] / oi_shift - 1

        # Amount ratio
        amt_ma = _group_rolling(df, "amount", n, "mean")
        derived[f"amount_ratio_{n}"] = df["amount"] / amt_ma

        # VWAP deviation
        amt_sum = _group_rolling(df, "amount", n, "sum")
        vol_sum = _group_rolling(df, "volume", n, "sum")
        vwap = amt_sum / vol_sum
        derived[f"vwap_dev_{n}"] = df["close"] / vwap - 1

    # Add all derived features to dataframe
    for k, s in derived.items():
        df[k] = s

    # Rolling stats + diff for each derived feature
    stat_cols = list(derived.keys())
    for k in stat_cols:
        df[f"{k}_mean20"] = _group_rolling(df, k, STAT_WINDOW, "mean")
        df[f"{k}_std20"] = _group_rolling(df, k, STAT_WINDOW, "std")
        df[f"{k}_diff1"] = df.groupby("symbol")[k].diff(1)

    # Cast everything numeric to float32 (except date/symbol)
    for c in df.columns:
        if c in ("date", "symbol"):
            continue
        df[c] = df[c].astype("float32")

    return df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Normalization
# ────────────────────────────────────────────────────────────────────────────


def xsec_zscore(df: pd.DataFrame, exclude: tuple[str, ...] = ("date", "symbol")) -> pd.DataFrame:
    """Apply cross-sectional z-score normalization per day.

    For each date, standardizes numeric features to zero mean and unit std
    across all symbols.

    Args:
        df: DataFrame with 'date' column and numeric features
        exclude: Column names to skip normalization (typically date, symbol)

    Returns:
        DataFrame with numeric columns normalized per day
    """
    df = df.copy()

    # Identify numeric columns to normalize
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Group by date and apply z-score
    for col in numeric_cols:
        mean = df.groupby("date")[col].transform("mean")
        std = df.groupby("date")[col].transform(lambda x: x.std(ddof=0))
        df[col] = ((df[col] - mean) / std).fillna(0.0)

    return df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Labeling
# ────────────────────────────────────────────────────────────────────────────


def add_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add forward return and cross-sectional rank label.

    Computes:
    - ret_Nd: (close_t+N / close_t) - 1
    - label: cross-sectional rank of ret_Nd within each day, scaled to [-1, 1]

    Args:
        df: DataFrame with 'date', 'symbol', 'close' columns
        horizon: Forward return horizon in days

    Returns:
        DataFrame with 'ret_Nd' and 'label' columns added
    """
    df = df.copy()

    # Sort by symbol and date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Compute forward return per symbol
    ret_col = f"ret_{horizon}d"
    df[ret_col] = df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1

    # Compute cross-sectional rank per date
    # Get count of non-null returns per date
    valid_counts = df.groupby("date")[ret_col].transform(lambda x: x.notna().sum())

    # Rank returns ascending (lowest return = rank 1)
    ranks = df.groupby("date")[ret_col].rank(method="average", na_option="keep")

    # Scale to [-1, 1]: rank 1 → -1, rank n → 1
    # When valid_counts <= 1, label = 0
    df["label"] = 0.0
    mask = valid_counts > 1
    df.loc[mask, "label"] = 2 * (ranks[mask] - 1) / (valid_counts[mask] - 1) - 1

    return df.reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────


def build_feature_frame(start: str, end: str, symbols: list[str] | None = None) -> pd.DataFrame:
    """Build complete feature frame with normalization and labels.

    Orchestrates the full pipeline:
    1. Load raw panel data
    2. Add engineered features
    3. Drop rows with NaN in mom_60 (longest lookback)
    4. Apply cross-sectional z-score normalization
    5. Add forward return labels

    Args:
        start: Start date in YYYYMMDD or YYYY-MM-DD format
        end: End date in YYYYMMDD or YYYY-MM-DD format
        symbols: List of symbols (if None, uses all commodity futures)

    Returns:
        DataFrame with complete feature set and labels
    """
    # Load symbols if not provided
    if symbols is None:
        symbols = list_commodity_symbols()

    # Load raw panel
    raw = load_raw_panel(symbols, start, end)

    # Add engineered features
    feat = add_engineered_features(raw)

    # Drop rows with NaN in mom_60 (longest lookback window)
    feat = feat.dropna(subset=["mom_60"]).reset_index(drop=True)

    # Apply cross-sectional normalization
    feat = xsec_zscore(feat, exclude=("date", "symbol"))

    # Add labels
    feat = add_label(feat, horizon=5)

    return feat


# ────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point for feature generation.

    Reads environment variables:
    - PANDA_DATA_START_DATE: Start date (YYYYMMDD or YYYY-MM-DD)
    - PANDA_DATA_END_DATE: End date (YYYYMMDD or YYYY-MM-DD)
    - PANDA_DATA_USERNAME: Panda data username
    - PANDA_DATA_PASSWORD: Panda data password

    Writes:
    - data/features.parquet: Complete feature frame
    """
    import os
    from pathlib import Path

    # Read environment variables
    start = _get_env("PANDA_DATA_START_DATE")
    end = _get_env("PANDA_DATA_END_DATE")

    # Build features
    print(f"Building features from {start} to {end}...")
    df = build_feature_frame(start, end)

    # Write to production/data/features.parquet
    data_dir = Path(__file__).parent.parent.parent / "dl-transformer-multiasset-production" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "features.parquet"

    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
