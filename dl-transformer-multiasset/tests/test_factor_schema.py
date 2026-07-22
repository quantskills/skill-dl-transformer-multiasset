"""Tests for factor.py: inference, scoring, and 12-column schema"""
import pandas as pd
import numpy as np
from scripts.factor import score_and_signal, emit_factor_table


def test_score_and_signal_boundaries():
    """Test that score_and_signal correctly assigns buy/sell/hold signals at quantile boundaries."""
    # Create synthetic predictions with known distribution
    # 20 symbols, 2 dates -> 40 rows
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    symbols = [f"S{i:02d}" for i in range(20)]

    data = []
    for date in dates:
        for i, sym in enumerate(symbols):
            data.append({
                "trade_date": date,
                "asset_type": "commodity",
                "symbol": sym,
                "factor_value": float(i),  # 0 to 19, evenly spaced
            })

    df = pd.DataFrame(data)

    # Apply scoring with 10% quantiles (buy top 10%, sell bottom 10%)
    result = score_and_signal(df, buy_q=0.1, sell_q=0.1)

    # Check per-day: for each date, top 2 should be "buy", bottom 2 "sell", rest "hold"
    for date in dates:
        day_df = result[result["trade_date"] == date]

        # Top 2 (factor_value >= 18)
        buy_signals = day_df[day_df["signal"] == "buy"]
        assert len(buy_signals) == 2, f"Expected 2 buy signals for {date}, got {len(buy_signals)}"
        assert buy_signals["factor_value"].min() >= 18.0

        # Bottom 2 (factor_value <= 1)
        sell_signals = day_df[day_df["signal"] == "sell"]
        assert len(sell_signals) == 2, f"Expected 2 sell signals for {date}, got {len(sell_signals)}"
        assert sell_signals["factor_value"].max() <= 1.0

        # Middle 16 should be "hold"
        hold_signals = day_df[day_df["signal"] == "hold"]
        assert len(hold_signals) == 16, f"Expected 16 hold signals for {date}, got {len(hold_signals)}"

        # Check rank is correct (0 to 19)
        assert day_df["rank"].min() == 0
        assert day_df["rank"].max() == 19

        # Check score is normalized to [0, 1]
        assert day_df["score"].min() >= 0.0
        assert day_df["score"].max() <= 1.0


def test_emit_factor_table_schema():
    """Test that emit_factor_table produces exact 12-column order and correct metadata."""
    # Create minimal scored dataframe
    scored = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "asset_type": ["commodity", "commodity"],
        "symbol": ["AG", "AU"],
        "factor_value": [0.5, -0.3],
        "score": [0.7, 0.3],
        "rank": [1, 0],
        "signal": ["buy", "sell"],
    })

    update_time = pd.Timestamp("2024-01-02 10:00:00")

    result = emit_factor_table(scored, update_time)

    # Check exact column order
    expected_cols = [
        "trade_date", "asset_type", "symbol", "factor_id", "factor_name",
        "factor_value", "score", "rank", "signal", "confidence",
        "data_version", "update_time"
    ]
    assert list(result.columns) == expected_cols, f"Column order mismatch: {list(result.columns)}"

    # Check metadata columns
    assert (result["factor_id"] == "dl_transformer_multiasset").all()
    assert (result["factor_name"] == "Deep Learning Transformer Multi-Asset").all()
    assert (result["confidence"] == 1.0).all()
    assert (result["data_version"] == "v1").all()
    assert (result["update_time"] == update_time).all()

    # Check data types
    assert result["trade_date"].dtype == "datetime64[ns]"
    assert result["update_time"].dtype == "datetime64[ns]"
    assert result["asset_type"].dtype == object
    assert result["symbol"].dtype == object
    assert result["factor_id"].dtype == object
    assert result["factor_name"].dtype == object
    assert result["signal"].dtype == object
    assert result["data_version"].dtype == object

    # Numeric columns
    assert pd.api.types.is_float_dtype(result["factor_value"])
    assert pd.api.types.is_float_dtype(result["score"])
    assert pd.api.types.is_integer_dtype(result["rank"])
    assert pd.api.types.is_float_dtype(result["confidence"])
