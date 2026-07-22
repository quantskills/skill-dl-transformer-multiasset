"""Tests for normalization and labeling functions in features.py."""
import numpy as np
import pandas as pd
from scripts import features


def _fake_panel(n_days: int = 100, symbols=("RB", "CU", "AU")) -> pd.DataFrame:
    """Create synthetic panel data for testing."""
    rows = []
    rng = np.random.RandomState(42)
    for sym in symbols:
        base = 100 + rng.randn() * 10
        prices = base + np.cumsum(rng.randn(n_days) * 2)
        for i, p in enumerate(prices):
            d = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
            rows.append({
                "date": d.strftime("%Y%m%d"),
                "symbol": sym,
                "close": p,
                "feat1": rng.randn() * 5,
                "feat2": rng.randn() * 10,
            })
    return pd.DataFrame(rows)


def test_xsec_zscore_zero_mean_unit_std():
    """xsec_zscore should produce zero mean and unit std per day."""
    df = _fake_panel()
    out = features.xsec_zscore(df, exclude=("date", "symbol"))

    # Group by date and check mean and std
    for date in out["date"].unique():
        day_df = out[out["date"] == date]
        for col in ["close", "feat1", "feat2"]:
            mean_val = day_df[col].mean()
            std_val = day_df[col].std(ddof=0)  # Population std
            assert abs(mean_val) < 1e-6, f"{col} mean {mean_val} not near zero on {date}"
            assert abs(std_val - 1.0) < 1e-6, f"{col} std {std_val} not near 1.0 on {date}"


def test_add_label_range():
    """add_label should produce labels in [-1, 1]."""
    df = _fake_panel()
    out = features.add_label(df, horizon=5)

    # Check ret_5d exists
    assert "ret_5d" in out.columns

    # Check label exists and is in range
    assert "label" in out.columns
    non_null = out["label"].dropna()
    assert non_null.min() >= -1.0, f"label min {non_null.min()} < -1"
    assert non_null.max() <= 1.0, f"label max {non_null.max()} > 1"
