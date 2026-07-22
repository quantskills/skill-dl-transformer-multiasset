"""Query interface for database.parquet.

This module provides:
- query(db_path, start, end, symbols, signals): Filter database.parquet
- main(): CLI with argparse

Usage:
    python query.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--symbols SYM1,SYM2] [--signals buy,sell,hold]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def query(
    db_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    signals: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filter database.parquet by start/end/symbols/signals.

    Args:
        db_path: Path to database.parquet file
        start: Start date (inclusive, YYYY-MM-DD format)
        end: End date (inclusive, YYYY-MM-DD format)
        symbols: List of symbols to filter (e.g., ["000001.SZ", "IF2401"])
        signals: List of signals to filter (e.g., ["buy", "sell"])

    Returns:
        Filtered DataFrame with 12-column schema

    Raises:
        FileNotFoundError: If database.parquet does not exist
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Load database
    df = pd.read_parquet(db_path)

    # Apply filters
    if start is not None:
        df = df[df["trade_date"] >= start]

    if end is not None:
        df = df[df["trade_date"] <= end]

    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]

    if signals is not None:
        df = df[df["signal"].isin(signals)]

    return df.reset_index(drop=True)


def main() -> None:
    """CLI entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Query database.parquet with optional filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query all data
  python query.py

  # Filter by date range
  python query.py --start 2024-01-01 --end 2024-12-31

  # Filter by symbols
  python query.py --symbols 000001.SZ,IF2401,600000.SH

  # Filter by signals
  python query.py --signals buy,sell

  # Combined filters
  python query.py --start 2024-01-01 --symbols 000001.SZ --signals buy
        """,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="../database.parquet",
        help="Path to database.parquet (default: ../database.parquet)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (e.g., 000001.SZ,IF2401)",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default=None,
        help="Comma-separated list of signals (e.g., buy,sell,hold)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for filtered results (default: print to stdout)",
    )

    args = parser.parse_args()

    # Parse comma-separated lists
    symbols = args.symbols.split(",") if args.symbols else None
    signals = args.signals.split(",") if args.signals else None

    # Query database
    result = query(
        db_path=args.db_path,
        start=args.start,
        end=args.end,
        symbols=symbols,
        signals=signals,
    )

    # Output
    if args.output:
        result.to_parquet(args.output, index=False)
        print(f"Filtered {len(result)} rows → {args.output}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
