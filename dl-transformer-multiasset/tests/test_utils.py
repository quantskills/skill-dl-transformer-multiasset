"""Tests for utils.py - constants and helper functions."""
import os
import pytest


def test_constants():
    """Verify all 9 constants are defined correctly."""
    from scripts import utils

    assert utils.FACTOR_ID == "DLTX"
    assert utils.FACTOR_NAME == "Transformer多资产联合建模"
    assert utils.DATA_VERSION == "real-v1"
    assert utils.ASSET_TYPE == "future"
    assert utils.BUY_QUANTILE == 0.1
    assert utils.SELL_QUANTILE == 0.1
    assert utils.LOOKBACK == 60
    assert utils.HORIZON == 5
    assert utils.SEED == 42


def test_date_conversion():
    """Test date format conversions in both directions."""
    from scripts import utils

    # Test _date_to_yyyymmdd
    assert utils._date_to_yyyymmdd("2024-01-15") == "20240115"
    assert utils._date_to_yyyymmdd("20240115") == "20240115"

    # Test _date_to_iso
    assert utils._date_to_iso("20240115") == "2024-01-15"
    assert utils._date_to_iso("2024-01-15") == "2024-01-15"

    # Test invalid input raises ValueError
    with pytest.raises(ValueError):
        utils._date_to_yyyymmdd("invalid")

    with pytest.raises(ValueError):
        utils._date_to_iso("invalid")


def test_get_env_missing():
    """Test that _get_env raises RuntimeError with env var name when missing."""
    from scripts import utils

    # Ensure the test env var doesn't exist
    test_var = "NONEXISTENT_TEST_VAR_12345"
    if test_var in os.environ:
        del os.environ[test_var]

    with pytest.raises(RuntimeError) as exc_info:
        utils._get_env(test_var)

    assert test_var in str(exc_info.value)


def test_batched():
    """Test _batched generator yields correct chunks."""
    from scripts import utils

    # Test basic batching
    result = list(utils._batched([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

    # Test exact division
    result = list(utils._batched([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

    # Test single batch
    result = list(utils._batched([1, 2, 3], 5))
    assert result == [[1, 2, 3]]

    # Test empty list
    result = list(utils._batched([], 2))
    assert result == []


def test_set_all_seeds():
    """Test set_all_seeds runs without errors."""
    from scripts import utils

    # Call twice to ensure it's idempotent and doesn't crash
    utils.set_all_seeds(42)
    utils.set_all_seeds(123)

    # Verify seeds are actually set
    import random
    import numpy as np

    utils.set_all_seeds(42)
    r1 = random.random()
    n1 = np.random.random()

    utils.set_all_seeds(42)
    r2 = random.random()
    n2 = np.random.random()

    assert r1 == r2, "Python random seed not working"
    assert n1 == n2, "NumPy random seed not working"
