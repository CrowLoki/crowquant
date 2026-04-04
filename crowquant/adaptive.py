"""Adaptive quantization with outlier detection."""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .core import (
    WHTransform, CrowQuantBlock, pack_bits, unpack_bits,
    quantize, dequantize,
)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive quantization."""
    n_bits: int = 4
    outlier_threshold: float = 3.0  # std devs for outlier detection
    use_wht: bool = True
    outlier_bits: int = 16  # float16 for outliers


class AdaptiveQuantizer:
    """Quantizer that detects and preserves outlier dimensions.

    High-magnitude channels (outliers) get stored at float16 precision
    while the bulk of dimensions use low-bit quantization. This preserves
    the information that matters most for similarity search.
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()

    def _detect_outliers(self, x: np.ndarray) -> np.ndarray:
        """Return boolean mask of outlier dimensions."""
        abs_vals = np.abs(x)
        mean = abs_vals.mean()
        std = abs_vals.std()
        if std < 1e-10:
            return np.zeros(len(x), dtype=bool)
        threshold = mean + self.config.outlier_threshold * std
        return abs_vals > threshold

    def quantize(self, x: np.ndarray) -> CrowQuantBlock:
        """Quantize with adaptive outlier handling.

        Outlier channels are stored separately at float16, while the
        remaining channels use n-bit uniform quantization.
        """
        x = np.asarray(x, dtype=np.float64)
        dim = len(x)

        if self.config.use_wht:
            coeffs, orig_len = WHTransform.forward(x)
        else:
            coeffs = x.copy()
            orig_len = dim

        # detect outliers in transform domain
        outlier_mask = self._detect_outliers(coeffs)
        n_outliers = outlier_mask.sum()

        if n_outliers > 0 and n_outliers < len(coeffs) * 0.1:
            # store outliers at float16
            outlier_values = coeffs[outlier_mask].astype(np.float16).tobytes()
            outlier_mask_bytes = np.packbits(outlier_mask).tobytes()

            # zero out outliers before quantizing the rest
            coeffs_clean = coeffs.copy()
            coeffs_clean[outlier_mask] = 0.0
        else:
            # too many or no outliers -- skip outlier handling
            outlier_values = None
            outlier_mask_bytes = None
            coeffs_clean = coeffs

        # uniform quantize the non-outlier channels
        n_levels = 1 << self.config.n_bits
        vmin = coeffs_clean.min()
        vmax = coeffs_clean.max()

        if vmax == vmin:
            scale = 1.0
            zero_point = vmin
            indices = np.zeros(len(coeffs_clean), dtype=np.uint64)
        else:
            scale = (vmax - vmin) / (n_levels - 1)
            zero_point = vmin
            indices = np.round((coeffs_clean - zero_point) / scale).astype(np.uint64)
            indices = np.clip(indices, 0, n_levels - 1)

        packed = pack_bits(indices, self.config.n_bits)

        return CrowQuantBlock(
            n_bits=self.config.n_bits,
            dim=dim,
            scale=scale,
            zero_point=zero_point,
            packed=packed,
            outlier_mask=outlier_mask_bytes,
            outlier_values=outlier_values,
        )

    def dequantize(self, block: CrowQuantBlock) -> np.ndarray:
        """Reconstruct vector, merging outlier values back in."""
        if self.config.use_wht:
            n = 1
            while n < block.dim:
                n <<= 1
            count = n
        else:
            count = block.dim

        indices = unpack_bits(block.packed, block.n_bits, count)
        coeffs = indices.astype(np.float64) * block.scale + block.zero_point

        # restore outliers
        if block.outlier_mask is not None and block.outlier_values is not None:
            mask = np.unpackbits(
                np.frombuffer(block.outlier_mask, dtype=np.uint8)
            )[:count].astype(bool)
            outlier_vals = np.frombuffer(block.outlier_values, dtype=np.float16).astype(np.float64)
            coeffs[mask] = outlier_vals

        if self.config.use_wht:
            return WHTransform.inverse(coeffs, block.dim)
        else:
            return coeffs[:block.dim]
