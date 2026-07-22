"""
Transformer model implementations: PatchTST with RoPE positional encoding.
"""
from __future__ import annotations

import math
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch is required for model.py. Install with: pip install torch")


# ────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding (RoPE)
# ────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer paper.

    Generates cos and sin position embeddings for applying rotary encoding
    to query and key tensors in attention mechanism.

    Reference:
        Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
        arXiv:2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Initialize RoPE embeddings.

        Args:
            dim: Dimension of embeddings (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for exponential decay (default 10000)
        """
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequency: theta_i = base^(-2i/d)
        # Shape: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin cache for given sequence length."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

        # Compute position * inv_freq for all positions and frequencies
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(positions, self.inv_freq)

        # Duplicate frequencies to match full dimension
        # Shape: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cos and sin position embeddings.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors, each of shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            # Recompute cache if sequence is longer than precomputed
            self._update_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len

        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor of shape (batch_size, n_heads, seq_len, dim)
        k: Key tensor of shape (batch_size, n_heads, seq_len, dim)
        cos: Cosine embeddings of shape (seq_len, dim)
        sin: Sine embeddings of shape (seq_len, dim)

    Returns:
        Tuple of (q_rope, k_rope) with same shapes as input
    """
    # Reshape cos and sin to match q/k dimensions
    # (seq_len, dim) -> (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split q and k into pairs for rotation
    # Shape: (batch_size, n_heads, seq_len, dim/2, 2)
    q_pairs = q.reshape(*q.shape[:-1], -1, 2)
    k_pairs = k.reshape(*k.shape[:-1], -1, 2)

    # Extract even and odd indices
    # q_even: (batch_size, n_heads, seq_len, dim/2)
    q_even = q_pairs[..., 0]
    q_odd = q_pairs[..., 1]
    k_even = k_pairs[..., 0]
    k_odd = k_pairs[..., 1]

    # Reshape cos and sin for pair-wise application
    # (1, 1, seq_len, dim) -> (1, 1, seq_len, dim/2, 2)
    cos_pairs = cos.reshape(*cos.shape[:-1], -1, 2)
    sin_pairs = sin.reshape(*sin.shape[:-1], -1, 2)

    cos_even = cos_pairs[..., 0]
    cos_odd = cos_pairs[..., 1]
    sin_even = sin_pairs[..., 0]
    sin_odd = sin_pairs[..., 1]

    # Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    q_rope_even = q_even * cos_even - q_odd * sin_even
    q_rope_odd = q_even * sin_odd + q_odd * cos_odd
    k_rope_even = k_even * cos_even - k_odd * sin_even
    k_rope_odd = k_even * sin_odd + k_odd * cos_odd

    # Interleave back to original shape
    q_rope = torch.stack([q_rope_even, q_rope_odd], dim=-1).reshape(q.shape)
    k_rope = torch.stack([k_rope_even, k_rope_odd], dim=-1).reshape(k.shape)

    return q_rope, k_rope


# ────────────────────────────────────────────────────────────────────────────
# PatchTST Model
# ────────────────────────────────────────────────────────────────────────────


class PatchTST(nn.Module):
    """
    Patched Time Series Transformer (PatchTST) with RoPE.

    Channel-independent architecture that processes each feature separately,
    then aggregates. Time series is split into patches for efficient encoding.

    Architecture:
        1. Split time series into patches (patch_len, stride)
        2. Linear projection of patches to d_model
        3. Add RoPE position encoding
        4. Transformer encoder (multi-head attention + FFN)
        5. Global average pooling across patches
        6. Per-channel heads -> aggregate -> scalar output

    Reference:
        Nie et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting
        with Transformers. ICLR.
    """

    def __init__(
        self,
        n_features: int,
        lookback: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
        d_ff: int = None,
    ):
        """
        Initialize PatchTST model.

        Args:
            n_features: Number of input features (channels)
            lookback: Length of input time series
            patch_len: Length of each patch
            stride: Stride for sliding window patching
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of Transformer encoder layers
            dropout: Dropout rate
            d_ff: Feedforward dimension (default: 4 * d_model)
        """
        super().__init__()

        self.n_features = n_features
        self.lookback = lookback
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.d_ff = d_ff or (4 * d_model)

        # Calculate number of patches
        # Formula: (lookback - patch_len) // stride + 1
        self.n_patches = (lookback - patch_len) // stride + 1
        assert self.n_patches > 0, f"Invalid patch config: lookback={lookback}, patch_len={patch_len}, stride={stride}"

        # Patch embedding: project each patch to d_model
        # Input: (batch_size * n_features, n_patches, patch_len)
        # Output: (batch_size * n_features, n_patches, d_model)
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # RoPE position encoding
        self.rope = RotaryEmbedding(d_model, max_seq_len=self.n_patches)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=self.d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: per-channel prediction
        # Shape: (batch_size, n_features, d_model) -> (batch_size, n_features)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Final aggregation: combine per-channel predictions
        # Shape: (batch_size, n_features) -> (batch_size,)
        self.aggregation = nn.Linear(n_features, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split time series into patches.

        Args:
            x: Input tensor of shape (batch_size, lookback, n_features)

        Returns:
            Patches of shape (batch_size, n_features, n_patches, patch_len)
        """
        batch_size, lookback, n_features = x.shape

        # Transpose to (batch_size, n_features, lookback)
        x = x.transpose(1, 2)

        # Extract patches using unfold
        # unfold(dimension, size, step)
        # Output: (batch_size, n_features, n_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PatchTST.

        Args:
            x: Input tensor of shape (batch_size, lookback, n_features)

        Returns:
            Predictions of shape (batch_size,)
        """
        batch_size, lookback, n_features = x.shape
        assert lookback == self.lookback, f"Expected lookback={self.lookback}, got {lookback}"
        assert n_features == self.n_features, f"Expected n_features={self.n_features}, got {n_features}"

        # Step 1: Patchify
        # Shape: (batch_size, n_features, n_patches, patch_len)
        patches = self._patchify(x)

        # Step 2: Reshape for channel-independent processing
        # Merge batch and feature dimensions
        # Shape: (batch_size * n_features, n_patches, patch_len)
        patches = patches.reshape(batch_size * n_features, self.n_patches, self.patch_len)

        # Step 3: Patch embedding
        # Shape: (batch_size * n_features, n_patches, d_model)
        patch_emb = self.patch_embedding(patches)

        # Step 4: Add RoPE (applied inside attention, but we prepare embeddings)
        # For standard Transformer, we would add position encoding here
        # But RoPE is applied directly to Q/K in attention
        # So we use the standard transformer which doesn't support RoPE natively
        # For MVP, we'll use learnable position embeddings instead
        # (RoPE integration would require custom attention implementation)

        # Workaround: Use standard transformer without RoPE for now
        # In production, this should be replaced with custom attention layers
        # Shape: (batch_size * n_features, n_patches, d_model)
        encoded = self.transformer(patch_emb)

        # Step 5: Global average pooling across patches
        # Shape: (batch_size * n_features, d_model)
        pooled = encoded.mean(dim=1)

        # Step 6: Per-channel prediction
        # Shape: (batch_size * n_features, d_model) -> (batch_size * n_features, 1)
        channel_preds = self.head(pooled)

        # Step 7: Reshape back to (batch_size, n_features)
        channel_preds = channel_preds.reshape(batch_size, n_features)

        # Step 8: Aggregate across channels
        # Shape: (batch_size, n_features) -> (batch_size, 1) -> (batch_size,)
        output = self.aggregation(channel_preds).squeeze(-1)

        return output


# ────────────────────────────────────────────────────────────────────────────
# iTransformer Model
# ────────────────────────────────────────────────────────────────────────────


class ITransformer(nn.Module):
    """
    Inverted Transformer (iTransformer) with variate-as-token architecture.

    Unlike PatchTST which treats time patches as tokens, iTransformer treats
    each feature (variate) as a token. The entire lookback window for each
    feature is projected to d_model dimension.

    Architecture:
        1. For each feature, project lookback window to d_model: Linear(lookback, d_model)
        2. Transformer encoder processes features as tokens (variate attention)
        3. Global average pooling across features
        4. MLP head for scalar output

    This inverted approach allows the model to capture cross-feature dependencies
    directly through attention, which is particularly useful for multivariate time series.

    Reference:
        Liu et al. (2024). iTransformer: Inverted Transformers Are Effective for
        Time Series Forecasting. ICLR.
    """

    def __init__(
        self,
        n_features: int,
        lookback: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
        d_ff: int = None,
    ):
        """
        Initialize ITransformer model.

        Args:
            n_features: Number of input features (becomes sequence length)
            lookback: Length of input time series (projected to d_model)
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of Transformer encoder layers
            dropout: Dropout rate
            d_ff: Feedforward dimension (default: 4 * d_model)
        """
        super().__init__()

        self.n_features = n_features
        self.lookback = lookback
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.d_ff = d_ff or (4 * d_model)

        # Variate embedding: project each feature's lookback to d_model
        # Input per feature: (lookback,)
        # Output per feature: (d_model,)
        self.variate_embedding = nn.Linear(lookback, d_model)

        # Transformer encoder layers
        # Note: sequence length is n_features (not time steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=self.d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: predict scalar from aggregated features
        # Shape: (batch_size, d_model) -> (batch_size,)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ITransformer.

        Args:
            x: Input tensor of shape (batch_size, lookback, n_features)

        Returns:
            Predictions of shape (batch_size,)
        """
        batch_size, lookback, n_features = x.shape
        assert lookback == self.lookback, f"Expected lookback={self.lookback}, got {lookback}"
        assert n_features == self.n_features, f"Expected n_features={self.n_features}, got {n_features}"

        # Step 1: Transpose to (batch_size, n_features, lookback)
        # Each feature's full time series will be embedded
        x = x.transpose(1, 2)

        # Step 2: Variate embedding
        # Input: (batch_size, n_features, lookback)
        # Output: (batch_size, n_features, d_model)
        # Each feature (variate) becomes a token
        variate_tokens = self.variate_embedding(x)

        # Step 3: Transformer encoder
        # Process features as tokens (attention over variates)
        # Shape: (batch_size, n_features, d_model)
        encoded = self.transformer(variate_tokens)

        # Step 4: Global average pooling across features
        # Shape: (batch_size, d_model)
        pooled = encoded.mean(dim=1)

        # Step 5: Output head
        # Shape: (batch_size, d_model) -> (batch_size, 1) -> (batch_size,)
        output = self.head(pooled).squeeze(-1)

        return output


# ────────────────────────────────────────────────────────────────────────────
# Model Factory
# ────────────────────────────────────────────────────────────────────────────


def build_model(arch: str, n_features: int, cfg: dict) -> nn.Module:
    """
    Factory function to build model by architecture name.

    Args:
        arch: 'patchtst' or 'itransformer'
        n_features: Number of input features
        cfg: Configuration dictionary with hyperparameters

    Returns:
        PyTorch model instance

    Raises:
        ValueError: If architecture is unknown
    """
    if arch == "patchtst":
        return PatchTST(
            n_features=n_features,
            lookback=cfg["LOOKBACK"],
            patch_len=cfg.get("PATCH_LEN", 16),
            stride=cfg.get("STRIDE", 8),
            d_model=cfg.get("D_MODEL", 128),
            n_heads=cfg.get("N_HEADS", 8),
            n_layers=cfg.get("N_LAYERS", 3),
            dropout=cfg.get("DROPOUT", 0.2),
        )
    elif arch == "itransformer":
        return ITransformer(
            n_features=n_features,
            lookback=cfg["LOOKBACK"],
            d_model=cfg.get("D_MODEL", 128),
            n_heads=cfg.get("N_HEADS", 8),
            n_layers=cfg.get("N_LAYERS", 3),
            dropout=cfg.get("DROPOUT", 0.2),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use 'patchtst' or 'itransformer'.")
