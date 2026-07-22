"""Tests for validate.py - 6 validation checks."""
import pandas as pd
import pytest
import tempfile
import json
from pathlib import Path


def test_required_fields_pass():
    """Test that a valid factor DataFrame passes required field checks."""
    from scripts import validate

    # Create valid DataFrame with all 12 required fields
    df = pd.DataFrame({
        'trade_date': ['2024-01-15', '2024-01-16'],
        'asset_type': ['future', 'future'],
        'symbol': ['RB', 'CU'],
        'factor_id': ['DLTX', 'DLTX'],
        'factor_name': ['Transformer多资产联合建模', 'Transformer多资产联合建模'],
        'factor_value': [0.5, -0.3],
        'score': [75.0, 25.0],
        'rank': [1, 2],
        'signal': ['buy', 'sell'],
        'confidence': [0.75, 0.25],
        'data_version': ['real-v1', 'real-v1'],
        'update_time': ['2024-01-15T09:00:00Z', '2024-01-16T09:00:00Z']
    })

    # Should not raise any exception
    validate.check_required_fields(df)


def test_required_fields_fail_missing_column():
    """Test that missing required column raises ValueError."""
    from scripts import validate

    # Missing 'rank' column
    df = pd.DataFrame({
        'trade_date': ['2024-01-15'],
        'asset_type': ['future'],
        'symbol': ['RB'],
        'factor_id': ['DLTX'],
        'factor_name': ['Transformer多资产联合建模'],
        'factor_value': [0.5],
        'score': [75.0],
        'signal': ['buy'],
        'confidence': [0.75],
        'data_version': ['real-v1'],
        'update_time': ['2024-01-15T09:00:00Z']
    })

    with pytest.raises(ValueError, match="rank"):
        validate.check_required_fields(df)


def test_value_range_fail_score_over_100():
    """Test that score > 100 raises ValueError."""
    from scripts import validate

    df = pd.DataFrame({
        'score': [50.0, 150.0],  # 150.0 is invalid
        'confidence': [0.5, 0.8],
        'rank': [1, 2]
    })

    with pytest.raises(ValueError, match="score"):
        validate.check_value_range(df)


def test_signal_enum_fail():
    """Test that invalid signal value raises ValueError."""
    from scripts import validate

    df = pd.DataFrame({
        'signal': ['buy', 'invalid_signal', 'hold']
    })

    with pytest.raises(ValueError, match="signal"):
        validate.check_signal_enum(df)


def test_oos_slice():
    """Test out-of-sample slice splits at midpoint correctly."""
    from scripts import validate

    # Create DataFrame with 10 rows
    df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'value': range(10)
    })

    # Should not raise - both halves non-empty
    validate.check_out_of_sample_slice(df)

    # Test with too few rows
    df_small = pd.DataFrame({
        'trade_date': ['2024-01-01'],
        'value': [1]
    })

    with pytest.raises(ValueError, match="out-of-sample"):
        validate.check_out_of_sample_slice(df_small)
