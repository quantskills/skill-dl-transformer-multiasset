"""Tests for backtest.py research bucket functions."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest import (
    annualized_return,
    information_ratio,
    pearson_ic,
    rank_ic,
)


def test_pearson_ic_zero_std():
    """Pearson IC should return NaN when factor or return has zero std."""
    df = pd.DataFrame({
        "factor_value": [1.0, 1.0, 1.0],
        "forward_return": [0.01, 0.02, 0.03],
    })
    result = pearson_ic(df)
    assert pd.isna(result)


def test_rank_ic_perfect():
    """Rank IC should return 1.0 for perfect rank correlation."""
    df = pd.DataFrame({
        "factor_value": [1.0, 2.0, 3.0, 4.0],
        "forward_return": [0.01, 0.02, 0.03, 0.04],
    })
    result = rank_ic(df)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_annualized_return_empty():
    """Annualized return should return 0.0 for empty series."""
    returns = pd.Series([], dtype=float)
    result = annualized_return(returns)
    assert result == 0.0


def test_information_ratio_zero_std():
    """Information ratio should return 0.0 when std is zero."""
    returns = pd.Series([0.0, 0.0, 0.0])
    result = information_ratio(returns)
    assert result == 0.0
