"""Adaptive per-channel bit allocation for CrowQuant.

Inspired by the llama.cpp finding that embedding channels have wildly
different norm magnitudes (6x-182x disparity).  High-norm "outlier"
channels get more bits to preserve their disproportionate contribution
to dot products and distances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import (
    CrowQuantBlock,
    WHTransform,
    lloyd_max_centroids,
    pack_bits,
    quantize_scalar,
    dequantize_scalar,
    unpack_bits,
)


@dataclass
class ChannelProfile:
    """Per-channel statistics from analyzing a batch of vectors.

    Attributes
    ----------
    channel_rms : np.ndarray
        RMS norm per channel across the analyzed batch.
    outlier_mask : np.ndarray
        Boolean mask -- True for channels flagged as outliers.
    bits_per_channel : np.ndarray
        Recommended bit allocation per channel.
    mean_rms : float
        Mean of channel_rms.
    std_rms : float
        Std dev of channel_rms.
    n_outliers : int
        Number of outlier channels.
    """

    channel_rms: np.ndarray
    outlier_mask: np.ndarray
    bits_per_channel: np.ndarray
    mean_rms: float
    std_rms: float
    n_outliers: int

    @property
    def dim(self) -> int:
        """Vector dimension this profile covers."""
        return len(self.channel_rms)

    @property
    def avg_bits(self) -> float:
        """Average bits per channel (effective bit-width)."""
        return float(np.mean(self.bits_per_channel))


@dataclass
class AdaptiveQuantBlock:
    """Container for an adaptively-quantized vector.

    Stores separate packed data for normal and outlier channels, plus
    the metadata needed to reconstruct.
    """

    normal_packed: bytes
    outlier_packed: bytes
    normal_scale: float
    normal_zero: float
    outlier_scale: float
    outlier_zero: float
    default_bits: int
    outlier_bits: int
    dim: int
    seed: int
    padded_dim: int
    outlier_indices: np.ndarray
    normal_indices: np.ndarray

    @property
    def compressed_size(self) -> int:
        """Total compressed size in bytes."""
        overhead = 64  # metadata
        return len(self.normal_packed) + len(self.outlier_packed) + overhead


class AdaptiveQuantizer:
    """Adaptive per-channel bit allocation quantizer.

    Channels whose RMS norm is more than ``outlier_threshold`` standard
    deviations above the mean get ``outlier_bits`` instead of
    ``default_bits``.  This preserves the high-impact channels that
    dominate dot products and distances.

    Parameters
    ----------
    default_bits : int
        Bits for normal channels (default 3).
    outlier_bits : int
        Bits for high-norm channels (default 8).
    outlier_threshold : float
        Number of std devs above mean RMS to flag as outlier (default 3.0).

    Examples
    --------
    >>> aq = AdaptiveQuantizer(default_bits=3, outlier_bits=8, outlier_threshold=2.0)
    >>> batch = np.random.randn(1000, 64)
    >>> batch[:, 0] *= 50  # make channel 0 an outlier
    >>> profile = aq.analyze(batch)
    >>> profile.outlier_mask[0]
    True
    >>> profile.bits_per_channel[0]
    8
    """

    def __init__(
        self,
        default_bits: int = 3,
        outlier_bits: int = 8,
        outlier_threshold: float = 3.0,
    ):
        self.default_bits = default_bits
        self.outlier_bits = outlier_bits
        self.outlier_threshold = outlier_threshold

    def analyze(self, vectors: np.ndarray) -> ChannelProfile:
        """Analyze a batch of vectors to determine per-channel statistics.

        Computes RMS norm per channel, identifies outliers, and returns
        a recommended bits-per-channel allocation.

        Parameters
        ----------
        vectors : np.ndarray
            2-D array of shape (n_vectors, dim).

        Returns
        -------
        ChannelProfile
            Per-channel analysis results.

        Examples
        --------
        >>> aq = AdaptiveQuantizer()
        >>> batch = np.random.randn(500, 32)
        >>> profile = aq.analyze(batch)
        >>> profile.dim
        32
        >>> profile.channel_rms.shape
        (32,)
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        channel_rms = np.sqrt(np.mean(vectors ** 2, axis=0))

        mean_rms = float(np.mean(channel_rms))
        std_rms = float(np.std(channel_rms))

        if std_rms < 1e-12:
            outlier_mask = np.zeros(vectors.shape[1], dtype=bool)
        else:
            outlier_mask = channel_rms > (mean_rms + self.outlier_threshold * std_rms)

        bits_per_channel = np.full(vectors.shape[1], self.default_bits, dtype=np.int32)
        bits_per_channel[outlier_mask] = self.outlier_bits

        return ChannelProfile(
            channel_rms=channel_rms,
            outlier_mask=outlier_mask,
            bits_per_channel=bits_per_channel,
            mean_rms=mean_rms,
            std_rms=std_rms,
            n_outliers=int(np.sum(outlier_mask)),
        )

    def quantize_adaptive(
        self,
        vector: np.ndarray,
        profile: ChannelProfile,
        seed: int = 42,
    ) -> AdaptiveQuantBlock:
        """Quantize with adaptive per-channel bit allocation.

        Outlier channels get more bits; normal channels get default_bits.

        Parameters
        ----------
        vector : np.ndarray
            1-D float vector to compress.
        profile : ChannelProfile
            Pre-computed channel statistics from analyze().
        seed : int
            WHT rotation seed.

        Returns
        -------
        AdaptiveQuantBlock
            Mixed-precision compressed representation.

        Examples
        --------
        >>> aq = AdaptiveQuantizer(default_bits=3, outlier_bits=8, outlier_threshold=2.0)
        >>> batch = np.random.randn(500, 64)
        >>> batch[:, 0] *= 50
        >>> profile = aq.analyze(batch)
        >>> v = batch[0]
        >>> blk = aq.quantize_adaptive(v, profile, seed=42)
        >>> blk.dim
        64
        """
        vector = np.asarray(vector, dtype=np.float64).ravel()
        orig_dim = len(vector)

        rotated = WHTransform.rotate(vector.copy(), seed=seed)
        padded_dim = len(rotated)

        outlier_idx = np.where(profile.outlier_mask)[0]
        normal_idx = np.where(~profile.outlier_mask)[0]

        # For padded dims beyond orig_dim, treat as normal
        if padded_dim > orig_dim:
            extra = np.arange(orig_dim, padded_dim)
            normal_idx = np.concatenate([normal_idx, extra])

        normal_vals = rotated[normal_idx]
        outlier_vals = rotated[outlier_idx] if len(outlier_idx) > 0 else np.array([])

        if len(normal_vals) > 0:
            n_idx, n_scale, n_zero = quantize_scalar(normal_vals, self.default_bits)
            normal_packed = pack_bits(n_idx, self.default_bits)
        else:
            n_scale, n_zero = 1.0, 0.0
            normal_packed = b""

        if len(outlier_vals) > 0:
            o_idx, o_scale, o_zero = quantize_scalar(outlier_vals, self.outlier_bits)
            outlier_packed = pack_bits(o_idx, self.outlier_bits)
        else:
            o_scale, o_zero = 1.0, 0.0
            outlier_packed = b""

        return AdaptiveQuantBlock(
            normal_packed=normal_packed,
            outlier_packed=outlier_packed,
            normal_scale=n_scale,
            normal_zero=n_zero,
            outlier_scale=o_scale,
            outlier_zero=o_zero,
            default_bits=self.default_bits,
            outlier_bits=self.outlier_bits,
            dim=orig_dim,
            seed=seed,
            padded_dim=padded_dim,
            outlier_indices=outlier_idx,
            normal_indices=normal_idx,
        )

    def dequantize_adaptive(
        self,
        block: AdaptiveQuantBlock,
        profile: ChannelProfile,
    ) -> np.ndarray:
        """Dequantize an adaptive block back to a float vector.

        Parameters
        ----------
        block : AdaptiveQuantBlock
            Compressed block from quantize_adaptive.
        profile : ChannelProfile
            Same profile used during quantization.

        Returns
        -------
        np.ndarray
            Reconstructed float64 vector of length block.dim.

        Examples
        --------
        >>> aq = AdaptiveQuantizer(default_bits=3, outlier_bits=8, outlier_threshold=2.0)
        >>> batch = np.random.randn(500, 64)
        >>> batch[:, 0] *= 50
        >>> profile = aq.analyze(batch)
        >>> v = batch[0]
        >>> blk = aq.quantize_adaptive(v, profile)
        >>> r = aq.dequantize_adaptive(blk, profile)
        >>> r.shape
        (64,)
        """
        rotated = np.zeros(block.padded_dim, dtype=np.float64)

        if len(block.normal_indices) > 0:
            n_idx = unpack_bits(
                block.normal_packed, block.default_bits, len(block.normal_indices)
            )
            normal_vals = dequantize_scalar(
                n_idx, block.default_bits, block.normal_scale, block.normal_zero
            )
            rotated[block.normal_indices] = normal_vals

        if len(block.outlier_indices) > 0:
            o_idx = unpack_bits(
                block.outlier_packed, block.outlier_bits, len(block.outlier_indices)
            )
            outlier_vals = dequantize_scalar(
                o_idx, block.outlier_bits, block.outlier_scale, block.outlier_zero
            )
            rotated[block.outlier_indices] = outlier_vals

        return WHTransform.unrotate(rotated, block.dim, block.seed)
