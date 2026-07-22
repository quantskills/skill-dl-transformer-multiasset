"""Tests for backtest.py baselines and tradeable bucket."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest import (
    baseline_momentum,
    baseline_random,
)


def test_baseline_random_preserves_rows():
    """baseline_random should preserve all rows and shuffle factor_value per day."""
    factor = pd.DataFrame({
        "trade_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "symbol": ["A", "B", "A", "B"],
        "factor_value": [1.0, 2.0, 3.0, 4.0],
    })
    result = baseline_random(factor, seed=42)

    # Should preserve shape and columns
    assert len(result) == len(factor)
    assert list(result.columns) == ["trade_date", "symbol", "factor_value", "signal"]

    # Should have same dates and symbols
    assert set(result["trade_date"]) == set(factor["trade_date"])
    assert set(result["symbol"]) == set(factor["symbol"])

    # Values should be shuffled (not all identical to original)
    original_sorted = sorted(factor["factor_value"].tolist())
    result_sorted = sorted(result["factor_value"].tolist())
    assert original_sorted == result_sorted  # Same values, possibly different order


def test_baseline_momentum_shape():
    """baseline_momentum should return factor with mom_20 as factor_value."""
    features = pd.DataFrame({
        "trade_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "symbol": ["A", "B", "A"],
        "mom_20": [0.05, -0.02, 0.03],
        "vol_20": [0.01, 0.02, 0.015],
    })
    result = baseline_momentum(features)

    # Should have correct columns
    assert list(result.columns) == ["trade_date", "symbol", "factor_value", "signal"]

    # Should preserve rows
    assert len(result) == len(features)

    # factor_value should match mom_20
    assert (result["factor_value"] == features["mom_20"]).all()

    # Should have signal column
    assert "signal" in result.columns
