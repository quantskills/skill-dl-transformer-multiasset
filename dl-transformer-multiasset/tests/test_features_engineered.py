"""Tests for engineered features in features.py."""
import numpy as np
import pandas as pd
from scripts import features


def _fake_panel(n_days: int = 200, symbols=("RB", "CU")) -> pd.DataFrame:
    """Create synthetic panel data for testing."""
    rows = []
    rng = np.random.RandomState(0)
    for sym in symbols:
        base = 100 + rng.randn()
        prices = base + np.cumsum(rng.randn(n_days) * 0.5)
        for i, p in enumerate(prices):
            d = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
            rows.append({
                "date": d.strftime("%Y%m%d"),
                "symbol": sym,
                "open": p, "high": p + 0.5, "low": p - 0.5, "close": p,
                "volume": 1000 + rng.rand() * 100,
                "amount": 1e5 + rng.rand() * 1000,
                "open_interest": 500 + rng.rand() * 50,
            })
    return pd.DataFrame(rows)


def test_engineered_feature_columns():
    """Verify all engineered feature columns exist."""
    raw = _fake_panel()
    out = features.add_engineered_features(raw)
    derived = [
        "mom_5", "mom_10", "mom_20", "mom_60",
        "rev_5", "rev_20",
        "vol_5", "vol_20", "vol_60",
        "vol_ratio_5", "vol_ratio_20",
        "oi_change_5", "oi_change_20",
        "amount_ratio_5", "amount_ratio_20",
        "vwap_dev_5", "vwap_dev_20",
    ]
    for f in derived:
        assert f in out.columns, f"missing {f}"
        assert f"{f}_mean20" in out.columns
        assert f"{f}_std20" in out.columns
        assert f"{f}_diff1" in out.columns


def test_engineered_no_lookahead():
    """Truncating future rows must not change past feature values."""
    raw = _fake_panel()
    full = features.add_engineered_features(raw)
    half_raw = raw[raw["date"] <= "20200601"].copy()
    half = features.add_engineered_features(half_raw)
    merged = full.merge(half, on=["date", "symbol"], suffixes=("_full", "_half"))
    for f in ["mom_20", "vol_20", "vwap_dev_20"]:
        a = merged[f"{f}_full"].dropna()
        b = merged[f"{f}_half"].dropna()
        common = merged.dropna(subset=[f"{f}_full", f"{f}_half"])
        diff = (common[f"{f}_full"] - common[f"{f}_half"]).abs()
        assert (diff < 1e-6).all(), f"{f} leaked future info"


def test_engineered_float32():
    """Verify all numeric features are float32."""
    out = features.add_engineered_features(_fake_panel())
    assert out["mom_20"].dtype == np.float32
    assert out["vol_20_mean20"].dtype == np.float32
