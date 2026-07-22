"""TDD tests for query.py production interface."""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import pandas as pd
import pytest

# Add parent scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from query import query


@pytest.fixture
def sample_database():
    """Create a sample database.parquet for testing."""
    data = {
        "trade_date": [
            "2024-01-01", "2024-01-01", "2024-01-01",
            "2024-01-02", "2024-01-02", "2024-01-02",
            "2024-01-03", "2024-01-03", "2024-01-03",
        ],
        "asset_type": ["stock", "future", "stock"] * 3,
        "symbol": ["000001.SZ", "IF2401", "600000.SH"] * 3,
        "factor_id": ["dl_transformer_multiasset"] * 9,
        "factor_name": ["Deep Learning Transformer Multi-Asset"] * 9,
        "factor_value": [0.5, -0.3, 0.8, 0.2, -0.5, 0.9, 0.1, -0.1, 0.7],
        "score": [60.0, 40.0, 80.0, 55.0, 35.0, 85.0, 52.0, 48.0, 75.0],
        "rank": [2, 3, 1, 2, 3, 1, 2, 3, 1],
        "signal": ["buy", "sell", "buy", "hold", "sell", "buy", "hold", "hold", "buy"],
        "confidence": [1.0] * 9,
        "data_version": ["v1"] * 9,
        "update_time": ["2024-01-04T00:00:00"] * 9,
    }
    df = pd.DataFrame(data)

    # Create temp database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_path = f.name

    df.to_parquet(temp_path, index=False)

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def test_query_filters(sample_database):
    """Verify start/end/symbol/signal filtering."""
    # Test 1: Filter by start date
    result = query(sample_database, start="2024-01-02")
    assert len(result) == 6
    assert result["trade_date"].min() == "2024-01-02"

    # Test 2: Filter by end date
    result = query(sample_database, end="2024-01-02")
    assert len(result) == 6
    assert result["trade_date"].max() == "2024-01-02"

    # Test 3: Filter by start and end
    result = query(sample_database, start="2024-01-02", end="2024-01-02")
    assert len(result) == 3
    assert (result["trade_date"] == "2024-01-02").all()

    # Test 4: Filter by symbols
    result = query(sample_database, symbols=["000001.SZ", "IF2401"])
    assert len(result) == 6
    assert set(result["symbol"].unique()) == {"000001.SZ", "IF2401"}

    # Test 5: Filter by signals
    result = query(sample_database, signals=["buy"])
    assert len(result) == 4
    assert (result["signal"] == "buy").all()

    # Test 6: Combined filters
    result = query(
        sample_database,
        start="2024-01-02",
        end="2024-01-03",
        symbols=["000001.SZ", "600000.SH"],
        signals=["buy", "hold"]
    )
    # 2024-01-02: 000001.SZ=hold, 600000.SH=buy
    # 2024-01-03: 000001.SZ=hold, 600000.SH=buy
    # Total: 4 rows
    assert len(result) == 4
    assert result["trade_date"].isin(["2024-01-02", "2024-01-03"]).all()
    assert result["symbol"].isin(["000001.SZ", "600000.SH"]).all()
    assert result["signal"].isin(["buy", "hold"]).all()

    # Test 7: No filters (return all)
    result = query(sample_database)
    assert len(result) == 9
