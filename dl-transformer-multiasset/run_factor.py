#!/usr/bin/env python
"""
Adapter script to run factor inference with the actual checkpoint structure.
Bridges the gap between train.py's checkpoint format and factor.py's expectations.
"""
import sys
from pathlib import Path
import shutil
import tempfile
import pandas as pd

def prepare_fold_dirs(checkpoint_dir: Path) -> list[Path]:
    """
    Create temporary fold directories matching factor.py's expected structure.

    Expected by factor.py: fold_dirs each contain 'best_model.pt'
    Actual structure: checkpoints/fold_X_best.pth

    Returns:
        List of temporary directories, each containing a best_model.pt symlink
    """
    checkpoint_files = sorted(checkpoint_dir.glob("fold_*_best.pth"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    temp_dirs = []
    temp_base = Path(tempfile.mkdtemp(prefix="factor_folds_"))

    for i, ckpt_file in enumerate(checkpoint_files):
        # Create fold directory
        fold_dir = temp_base / f"fold_{i}"
        fold_dir.mkdir()

        # Create symlink with expected name
        target = fold_dir / "best_model.pt"
        target.symlink_to(ckpt_file.resolve())

        temp_dirs.append(fold_dir)
        print(f"Prepared fold {i}: {ckpt_file.name} -> {target}")

    return temp_dirs, temp_base


def main():
    # Paths
    feature_path = "../dl-transformer-multiasset-production/data/features.parquet"
    checkpoint_dir = Path("checkpoints")
    # Output to production data directory (parent/../production/data/)
    output_dir = Path(__file__).parent.parent / "dl-transformer-multiasset-production" / "data"

    # Validate inputs
    if not Path(feature_path).exists():
        print(f"Error: Feature file not found: {feature_path}")
        print("Run 'python -m scripts.features' first")
        sys.exit(1)

    if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("fold_*_best.pth")):
        print(f"Error: No trained checkpoints found in {checkpoint_dir}")
        print("Run 'python -m scripts.train' first")
        sys.exit(1)

    # Prepare fold directories
    print("Preparing checkpoint directories...")
    try:
        fold_dirs, temp_base = prepare_fold_dirs(checkpoint_dir)

        # Import and run factor inference
        print("\nRunning factor inference...")
        from scripts.factor import main as factor_main

        factor_main(
            feature_path=str(feature_path),
            output_dir=str(output_dir),
            fold_dirs=[str(d) for d in fold_dirs],
            device_pref="auto"
        )

        print(f"\n✓ Factor table written to {output_dir / 'database.parquet'}")
        print(f"  Production directory: {output_dir.resolve()}")

    finally:
        # Cleanup temporary directories
        if 'temp_base' in locals():
            print(f"\nCleaning up temporary directories...")
            shutil.rmtree(temp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
