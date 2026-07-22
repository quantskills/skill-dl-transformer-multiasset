"""Tests for features.py data loading functions."""
import pandas as pd
import pytest


def test_normalize_symbol_col():
    """Test contract symbol normalization to underlying symbol."""
    from scripts.features import _normalize_symbol_col

    # Test with contract codes
    df = pd.DataFrame(
        {
            "symbol": ["RB2410", "AU2412", "CU2501", "AL2503"],
            "price": [3500.0, 520.0, 74000.0, 19000.0],
        }
    )

    result = _normalize_symbol_col(df, src_col="symbol")

    # Should strip trailing digits
    expected_symbols = ["RB", "AU", "CU", "AL"]
    assert result["symbol"].tolist() == expected_symbols
    # Should preserve other columns
    assert result["price"].tolist() == [3500.0, 520.0, 74000.0, 19000.0]


def test_load_raw_panel_schema(monkeypatch):
    """Test load_raw_panel returns correct schema and types."""
    from scripts.features import load_raw_panel

    # Mock environment variables
    monkeypatch.setenv("PANDA_USERNAME", "test_user")
    monkeypatch.setenv("PANDA_PASSWORD", "test_pass")

    # Mock panda_data.init_token to avoid actual API calls
    def mock_init_token(**kwargs):
        pass

    monkeypatch.setattr("panda_data.init_token", mock_init_token)

    # Mock the batch fetch function to return fake data
    def mock_fetch_batch(symbols, start, end):
        return pd.DataFrame(
            {
                "date": ["20240101", "20240101", "20240102"],
                "symbol": ["RB", "AU", "RB"],
                "open": [3500.0, 520.0, 3510.0],
                "high": [3550.0, 525.0, 3560.0],
                "low": [3480.0, 518.0, 3490.0],
                "close": [3520.0, 522.0, 3540.0],
                "volume": [100000.0, 50000.0, 95000.0],
                "amount": [352000000.0, 26100000.0, 336300000.0],
                "open_interest": [80000.0, 30000.0, 78000.0],
            }
        )

    # Patch the internal batch fetch function
    monkeypatch.setattr("scripts.features._fetch_daily_post_batch", mock_fetch_batch)

    # Call load_raw_panel
    result = load_raw_panel(["RB", "AU"], "20240101", "20240102")

    # Verify schema
    expected_columns = [
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
    assert list(result.columns) == expected_columns

    # Verify types
    assert result["date"].dtype == object  # string
    assert result["symbol"].dtype == object  # string
    assert result["open"].dtype == float
    assert result["high"].dtype == float
    assert result["low"].dtype == float
    assert result["close"].dtype == float
    assert result["volume"].dtype == float
    assert result["amount"].dtype == float
    assert result["open_interest"].dtype == float

    # Verify shape
    assert len(result) == 3
    assert len(result.columns) == 9
