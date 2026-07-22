"""Shared constants and helper functions for Transformer Multi-Asset skill."""
from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Generator, TypeVar

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
FACTOR_ID = "DLTX"
FACTOR_NAME = "Transformer多资产联合建模"
DATA_VERSION = "real-v1"
ASSET_TYPE = "future"
BUY_QUANTILE = 0.1
SELL_QUANTILE = 0.1
LOOKBACK = 60
HORIZON = 5
SEED = 42

# ────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────


def _get_env(name: str) -> str:
    """Get environment variable or raise RuntimeError if missing.

    Args:
        name: Environment variable name

    Returns:
        Environment variable value

    Raises:
        RuntimeError: If environment variable is not set
    """
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


# ────────────────────────────────────────────────────────────────────────────
# Date Utilities
# ────────────────────────────────────────────────────────────────────────────


def _date_to_yyyymmdd(value: str) -> str:
    """Convert date to YYYYMMDD format.

    Args:
        value: Date string in YYYY-MM-DD or YYYYMMDD format

    Returns:
        Date string in YYYYMMDD format

    Raises:
        ValueError: If date format is invalid
    """
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        # YYYY-MM-DD format
        return value.replace("-", "")
    elif len(value) == 8 and value.isdigit():
        # YYYYMMDD format
        return value
    else:
        raise ValueError(f"Invalid date format: {value}. Expected YYYY-MM-DD or YYYYMMDD")


def _date_to_iso(value: str) -> str:
    """Convert date to ISO format (YYYY-MM-DD).

    Args:
        value: Date string in YYYYMMDD or YYYY-MM-DD format

    Returns:
        Date string in YYYY-MM-DD format

    Raises:
        ValueError: If date format is invalid
    """
    if len(value) == 8 and value.isdigit():
        # YYYYMMDD format
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    elif len(value) == 10 and value[4] == "-" and value[7] == "-":
        # Already in ISO format
        return value
    else:
        raise ValueError(f"Invalid date format: {value}. Expected YYYYMMDD or YYYY-MM-DD")


# ────────────────────────────────────────────────────────────────────────────
# Retry Logic
# ────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")


def _call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 6,
    base_wait: float = 10.0,
    **kwargs: Any,
) -> T:
    """Call function with retry logic for panda_data API errors.

    Handles:
    - 500010: Rate limit exceeded - retry with exponential backoff
    - 200004: Token expired - re-login and retry

    Args:
        fn: Function to call
        *args: Positional arguments to pass to function
        max_retries: Maximum number of retry attempts
        base_wait: Base wait time in seconds (doubled each retry)
        **kwargs: Keyword arguments to pass to function

    Returns:
        Function return value

    Raises:
        Exception: If all retries exhausted or non-retryable error
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Try to extract error code from panda_data ServiceError
            error_code = None
            if hasattr(e, "code"):
                error_code = str(e.code)
            elif "500010" in str(e) or "rate limit" in str(e).lower():
                error_code = "500010"
            elif "200004" in str(e) or "token expired" in str(e).lower():
                error_code = "200004"

            # Handle rate limit
            if error_code == "500010":
                if attempt < max_retries:
                    wait = base_wait * (2**attempt)
                    print(f"Rate limit hit, waiting {wait:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait)
                    continue
                else:
                    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries") from e

            # Handle token expiration
            elif error_code == "200004":
                if attempt < max_retries:
                    print(f"Token expired, re-initializing... (attempt {attempt + 1}/{max_retries})")
                    try:
                        import panda_data

                        panda_data.init_token()
                    except Exception as init_error:
                        print(f"Failed to re-initialize token: {init_error}")
                        raise
                    continue
                else:
                    raise RuntimeError(f"Token re-initialization failed after {max_retries} retries") from e

            # Non-retryable error
            else:
                raise


# ────────────────────────────────────────────────────────────────────────────
# Collections
# ────────────────────────────────────────────────────────────────────────────


def _batched(lst: list[T], n: int) -> Generator[list[T], None, None]:
    """Split list into chunks of size n.

    Args:
        lst: List to batch
        n: Batch size

    Yields:
        Batches of up to n elements
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ────────────────────────────────────────────────────────────────────────────
# Random Seeds
# ────────────────────────────────────────────────────────────────────────────


def set_all_seeds(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (if available)
    - PyTorch CUDA (if available)

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Enable deterministic operations in cuDNN
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            # cuDNN not available
            pass
    except ImportError:
        # PyTorch not available
        pass
