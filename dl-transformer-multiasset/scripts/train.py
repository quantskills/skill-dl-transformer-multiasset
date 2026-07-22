"""
Training setup: device selection, CPU guard, mixed precision, loss functions, hyperparameters
"""
import os
from typing import Tuple, Optional
from contextlib import contextmanager
import torch
import torch.nn as nn


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CFG = {
    # Model architecture
    "LOOKBACK": 40,         # Reduced from 60 to lower memory usage
    "PATCH_LEN": 10,        # Reduced from 16 for fewer patches
    "STRIDE": 5,            # Reduced from 8
    "D_MODEL": 64,          # Reduced from 128 (half the hidden dimension)
    "N_HEADS": 4,           # Reduced from 8 (fewer attention heads)
    "N_LAYERS": 2,          # Reduced from 3 (fewer transformer layers)
    "DROPOUT": 0.2,

    # Training
    "LR": 1e-4,
    "BATCH_SIZE": 32,       # Reduced from 64 to process less data per batch
    "MAX_EPOCHS": 50,
    "EARLY_STOP_PATIENCE": 10,
    "SEED": 42,

    # Loss
    "LOSS_ALPHA": 0.5  # alpha*rank_ic_loss + (1-alpha)*mse
}


# ============================================================================
# Device Selection
# ============================================================================

def pick_device(preference: str = "auto") -> torch.device:
    """
    Pick compute device with priority: cuda > mps > cpu

    Args:
        preference: "auto" (default), "cuda", "mps", or "cpu"

    Returns:
        torch.device instance

    Raises:
        RuntimeError: If forced device is not available
        ValueError: If preference is unknown
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    elif preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        return torch.device("mps")
    elif preference == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {preference}")


# ============================================================================
# CPU Training Guard
# ============================================================================

def guard_cpu_training(device: torch.device, n_samples: int, threshold: int = 10000) -> None:
    """
    Block CPU training on large datasets unless ALLOW_CPU_TRAIN=1

    Args:
        device: Compute device
        n_samples: Number of training samples
        threshold: Sample count threshold (default 10000)

    Raises:
        RuntimeError: If CPU + large dataset without ALLOW_CPU_TRAIN=1
    """
    if device.type == "cpu" and n_samples > threshold:
        if os.environ.get("ALLOW_CPU_TRAIN") != "1":
            raise RuntimeError(
                f"Training on CPU with {n_samples} samples may take hours. "
                f"If you really want to proceed, set ALLOW_CPU_TRAIN=1"
            )
        else:
            print(f"[WARNING] Training on CPU with {n_samples} samples. This may take a long time.")


# ============================================================================
# Mixed Precision Context
# ============================================================================

@contextmanager
def mixed_precision_context(device: torch.device):
    """
    Context manager for mixed precision training

    - CUDA: autocast + GradScaler
    - MPS: autocast only (no GradScaler)
    - CPU: no mixed precision

    Args:
        device: Compute device

    Yields:
        Tuple of (autocast_context, scaler or None)

    Usage:
        with mixed_precision_context(device) as (autocast_ctx, scaler):
            with autocast_ctx:
                pred = model(X)
                loss = compute_loss(pred, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    """
    if device.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast()
        scaler = torch.cuda.amp.GradScaler()
        yield autocast_ctx, scaler
    elif device.type == "mps":
        # MPS uses 'cpu' tag for autocast but runs on MPS
        autocast_ctx = torch.amp.autocast("cpu")
        yield autocast_ctx, None
    else:
        # CPU: no mixed precision
        from contextlib import nullcontext
        yield nullcontext(), None


# ============================================================================
# Loss Functions
# ============================================================================

def rank_ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Rank IC loss: negative Pearson correlation

    Args:
        pred: Predictions, shape (batch_size,) or (batch_size, 1)
        target: Ground truth, shape (batch_size,) or (batch_size, 1)

    Returns:
        Scalar loss (negative correlation)
    """
    # Flatten to 1D
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Compute Pearson correlation
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

    # Avoid division by zero
    correlation = numerator / (denominator + 1e-8)

    # Return negative correlation as loss
    return -correlation


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Combined loss: alpha * rank_ic_loss + (1 - alpha) * mse

    Args:
        pred: Predictions, shape (batch_size,) or (batch_size, 1)
        target: Ground truth, shape (batch_size,) or (batch_size, 1)
        alpha: Weight for rank IC loss (default 0.5)

    Returns:
        Scalar loss
    """
    rank_loss = rank_ic_loss(pred, target)

    # Flatten both for MSE to avoid broadcasting warning
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    mse_loss = nn.functional.mse_loss(pred_flat, target_flat)

    return alpha * rank_loss + (1 - alpha) * mse_loss


# ============================================================================
# Walk-Forward Training
# ============================================================================

class _SeqDataset(torch.utils.data.Dataset):
    """
    Sequence dataset that builds lookback windows per symbol.

    For each symbol, creates sliding windows of (X[lookback, n_features], y)
    where X is the feature matrix and y is the label.

    Args:
        df: Feature dataframe with columns [date, symbol, features..., label]
        lookback: Number of time steps in each window (default 60)
        history_df: Optional historical data to use for lookback context.
                   If provided, will use history + df for creating windows but only
                   label samples from df.
    """

    def __init__(self, df: "pd.DataFrame", lookback: int = 60, history_df: "pd.DataFrame" = None):
        import pandas as pd

        self.lookback = lookback
        self.samples = []

        # Get feature columns (exclude date, symbol, label)
        feature_cols = [c for c in df.columns if c not in ["date", "symbol", "label", "ret_5d", "open", "high", "low", "close", "volume", "amount", "open_interest"]]
        label_col = "label"

        # Combine history and current data if history provided
        if history_df is not None:
            full_df = pd.concat([history_df, df], ignore_index=True).sort_values("date")
            # Mark which rows are in the target period - convert to timestamp for comparison
            target_dates = set(df["date"].values)
        else:
            full_df = df
            target_dates = None

        # Group by symbol and create windows
        for symbol, group in full_df.groupby("symbol"):
            # Sort by date
            group = group.sort_values("date").reset_index(drop=True)

            # Extract features and labels as numpy arrays
            X_full = group[feature_cols].values  # (T, F)
            y_full = group[label_col].values  # (T,)
            dates = group["date"].values  # numpy datetime64 array

            # Create sliding windows
            for i in range(lookback, len(group)):
                # Only create samples where the target date is in our period
                if target_dates is None or dates[i] in target_dates:
                    X_window = X_full[i - lookback:i]  # (lookback, F)
                    y_target = y_full[i]  # scalar

                    self.samples.append((X_window, y_target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


def make_folds(
    feature_df: "pd.DataFrame",
    train_months: int = 18,
    val_months: int = 3,
    test_months: int = 3
) -> list:
    """
    Create walk-forward folds for cross-validation.

    For data from 2022-04 to 2024-12, generates folds like:
    - Fold 0: train=[2022-04, 2023-10), val=[2023-10, 2024-01), test=[2024-01, 2024-04)
    - Fold 1: train=[2022-07, 2024-01), val=[2024-01, 2024-04), test=[2024-04, 2024-07)
    - Fold 2: train=[2022-10, 2024-04), val=[2024-04, 2024-07), test=[2024-07, 2024-10)

    Args:
        feature_df: Feature dataframe with 'date' column
        train_months: Number of months for training window (default 18)
        val_months: Number of months for validation window (default 3)
        test_months: Number of months for test window (default 3)

    Returns:
        List of fold dicts, each containing:
            - fold_id: Fold index
            - train: Training DataFrame
            - val: Validation DataFrame
            - test: Test DataFrame
    """
    import pandas as pd

    # Convert date column to datetime if it's stored as string
    if feature_df["date"].dtype == object or feature_df["date"].dtype == "string":
        feature_df = feature_df.copy()
        feature_df["date"] = pd.to_datetime(feature_df["date"])

    folds = []

    # Define fold start dates based on actual data range (2022-04 to 2024-12)
    fold_starts = [
        pd.Timestamp("2022-04-01"),
        pd.Timestamp("2022-07-01"),
        pd.Timestamp("2022-10-01"),
    ]

    for fold_id, train_start_ts in enumerate(fold_starts):
        # Calculate timestamps
        train_end_ts = train_start_ts + pd.DateOffset(months=train_months)
        val_end_ts = train_end_ts + pd.DateOffset(months=val_months)
        test_end_ts = val_end_ts + pd.DateOffset(months=test_months)

        # Split data
        df_train = feature_df[(feature_df["date"] >= train_start_ts) & (feature_df["date"] < train_end_ts)].copy()
        df_val = feature_df[(feature_df["date"] >= train_end_ts) & (feature_df["date"] < val_end_ts)].copy()
        df_test = feature_df[(feature_df["date"] >= val_end_ts) & (feature_df["date"] < test_end_ts)].copy()

        folds.append({
            "fold_id": fold_id,
            "train": df_train,
            "val": df_val,
            "test": df_test
        })

    return folds


def train_one_fold(
    fold: dict,
    feature_df: "pd.DataFrame",
    cfg: dict,
    device: torch.device,
    arch: str,
    ckpt_dir: str
) -> dict:
    """
    Train one fold with early stopping on val RankIC.

    Args:
        fold: Fold dict with keys [fold_id, train, val, test]
        feature_df: Full feature DataFrame (for reference)
        cfg: Configuration dict with hyperparameters
        device: Torch device (cpu/cuda/mps)
        arch: Model architecture ('patchtst' or 'itransformer')
        ckpt_dir: Directory to save checkpoints

    Returns:
        Dict with keys:
            - fold_id: Fold index
            - best_epoch: Epoch with best val RankIC
            - best_val_rank_ic: Best validation RankIC
            - checkpoint_path: Path to saved checkpoint
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr
    from scripts.model import PatchTST, ITransformer

    # Extract hyperparameters
    lookback = cfg["LOOKBACK"]
    batch_size = cfg["BATCH_SIZE"]
    max_epochs = cfg["MAX_EPOCHS"]
    patience = cfg["EARLY_STOP_PATIENCE"]
    lr = cfg["LR"]
    seed = cfg["SEED"]
    loss_alpha = cfg["LOSS_ALPHA"]

    # Set seed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Create datasets with historical context for lookback
    train_dataset = _SeqDataset(fold["train"], lookback=lookback)
    # For validation and test, use training data as history for lookback context
    val_dataset = _SeqDataset(fold["val"], lookback=lookback, history_df=fold["train"])

    # Check CPU training guard
    guard_cpu_training(device, len(train_dataset))

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Get number of features (exclude date, symbol, label, and raw price columns)
    exclude_cols = ["date", "symbol", "label", "ret_5d", "open", "high", "low", "close", "volume", "amount", "open_interest"]
    feature_cols = [c for c in fold["train"].columns if c not in exclude_cols]
    n_features = len(feature_cols)

    # Build model
    if arch == "patchtst":
        model = PatchTST(
            n_features=n_features,
            lookback=lookback,
            patch_len=cfg["PATCH_LEN"],
            stride=cfg["STRIDE"],
            d_model=cfg["D_MODEL"],
            n_heads=cfg["N_HEADS"],
            n_layers=cfg["N_LAYERS"],
            dropout=cfg["DROPOUT"]
        )
    elif arch == "itransformer":
        model = ITransformer(
            n_features=n_features,
            lookback=lookback,
            d_model=cfg["D_MODEL"],
            n_heads=cfg["N_HEADS"],
            n_layers=cfg["N_LAYERS"],
            dropout=cfg["DROPOUT"]
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mixed precision setup
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    best_val_rank_ic = -float("inf")
    best_epoch = 0
    patience_counter = 0
    fold_id = fold["fold_id"]

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(X)
                    loss = combined_loss(pred, y, alpha=loss_alpha)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(X)
                loss = combined_loss(pred, y, alpha=loss_alpha)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(X)
                        loss = combined_loss(pred, y, alpha=loss_alpha)
                else:
                    pred = model(X)
                    loss = combined_loss(pred, y, alpha=loss_alpha)

                val_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        val_loss /= len(val_loader)

        # Compute Rank IC
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        val_rank_ic, _ = spearmanr(all_preds, all_targets)

        # Check for best model
        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = os.path.join(ckpt_dir, f"fold_{fold_id}_best.pth")
            checkpoint = {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_rank_ic": val_rank_ic,
                "cfg": cfg,
                "arch": arch
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1

        # Print progress
        print(f"Fold {fold_id} - Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, "
              f"val_rank_ic={val_rank_ic:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Fold {fold_id} - Early stopping at epoch {epoch}")
            break

    # Return result
    checkpoint_path = os.path.join(ckpt_dir, f"fold_{fold_id}_best.pth")
    return {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_rank_ic": best_val_rank_ic,
        "checkpoint_path": checkpoint_path
    }


def main():
    """
    Main training entry point: walk-forward 5-fold training.

    Reads:
        - MODEL_ARCH: 'patchtst' (default) or 'itransformer'
        - TRAIN_DEVICE: 'auto' (default), 'cuda', 'mps', or 'cpu'
        - Feature table from data/features.parquet

    Writes:
        - Checkpoints to checkpoints/fold_*_best.pth
    """
    import pandas as pd

    # Read environment variables
    arch = os.environ.get("MODEL_ARCH", "patchtst").lower()
    device_pref = os.environ.get("TRAIN_DEVICE", "auto").lower()

    # Pick device
    device = pick_device(device_pref)
    print(f"[INFO] Using device: {device}")

    # Load feature table from production/data/
    feature_path = "../dl-transformer-multiasset-production/data/features.parquet"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature table not found at {feature_path}. "
            f"Run 'python -m scripts.features' first."
        )

    feature_df = pd.read_parquet(feature_path)
    print(f"[INFO] Loaded {len(feature_df)} rows from {feature_path}")

    # Create folds
    folds = make_folds(feature_df)
    print(f"[INFO] Created {len(folds)} folds")

    # Create checkpoint directory
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Train each fold
    results = []
    for fold in folds:
        print(f"\n[INFO] Training fold {fold['fold_id']}...")
        result = train_one_fold(fold, feature_df, DEFAULT_CFG, device, arch, ckpt_dir)
        results.append(result)
        print(f"[INFO] Fold {result['fold_id']} completed: "
              f"best_epoch={result['best_epoch']}, "
              f"best_val_rank_ic={result['best_val_rank_ic']:.4f}")

    # Report summary
    print("\n[INFO] Training completed!")
    print("[INFO] Summary:")
    for result in results:
        print(f"  Fold {result['fold_id']}: "
              f"val_rank_ic={result['best_val_rank_ic']:.4f}, "
              f"checkpoint={result['checkpoint_path']}")

    # Find best fold
    best_result = max(results, key=lambda r: r["best_val_rank_ic"])
    print(f"\n[INFO] Best fold: {best_result['fold_id']} "
          f"(val_rank_ic={best_result['best_val_rank_ic']:.4f})")


if __name__ == "__main__":
    main()
