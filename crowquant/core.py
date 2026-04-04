"""Core quantization algorithms for CrowQuant.

Implements Walsh-Hadamard Transform, Lloyd-Max scalar quantization,
bit packing, and the main quantize/dequantize API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

class WHTransform:
    """Walsh-Hadamard Transform utilities for randomized rotation.

    The WHT decorrelates vector components so that scalar quantization
    produces lower error (TurboQuant insight).  For non-power-of-2
    dimensions the vector is zero-padded, transformed, then stripped.
    """

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the smallest power of 2 >= n.

        >>> WHTransform._next_pow2(5)
        8
        >>> WHTransform._next_pow2(8)
        8
        """
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()

    @staticmethod
    def wht(x: np.ndarray) -> np.ndarray:
        """In-place Walsh-Hadamard Transform, O(d log d).

        Operates on a 1-D float array whose length must be a power of 2.
        The result is normalised by 1/sqrt(d) so the transform is its own
        inverse (up to floating-point noise).

        Parameters
        ----------
        x : np.ndarray
            1-D float array with power-of-2 length.

        Returns
        -------
        np.ndarray
            The transformed array (same object, modified in place).

        Examples
        --------
        >>> v = np.array([1.0, 0.0, 0.0, 0.0])
        >>> WHTransform.wht(v)
        array([0.5, 0.5, 0.5, 0.5])
        """
        d = len(x)
        h = 1
        while h < d:
            for i in range(0, d, h * 2):
                for j in range(i, i + h):
                    a, b = x[j], x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
            h *= 2
        x /= np.sqrt(d)
        return x

    @staticmethod
    def random_signs(d: int, seed: int = 42) -> np.ndarray:
        """Generate a reproducible array of +1/-1 sign flips.

        Parameters
        ----------
        d : int
            Dimension (number of signs to generate).
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of shape (d,) with values in {-1, +1}.

        Examples
        --------
        >>> s = WHTransform.random_signs(4, seed=0)
        >>> set(s).issubset({-1.0, 1.0})
        True
        """
        rng = np.random.default_rng(seed)
        return rng.choice([-1.0, 1.0], size=d).astype(np.float64)

    @staticmethod
    def rotate(x: np.ndarray, seed: int = 42) -> np.ndarray:
        """Apply randomised WHT rotation (sign flip then WHT).

        For non-power-of-2 dimensions the vector is zero-padded, rotated,
        then the padded region is retained (needed for faithful unrotation).

        Parameters
        ----------
        x : np.ndarray
            1-D float vector of any positive length.
        seed : int
            Seed controlling the random sign flips.

        Returns
        -------
        np.ndarray
            Rotated vector (length may be > len(x) due to padding).

        Examples
        --------
        >>> v = np.array([1.0, 2.0, 3.0])
        >>> r = WHTransform.rotate(v, seed=7)
        >>> r.shape[0]  # padded to 4
        4
        """
        orig_d = len(x)
        d = WHTransform._next_pow2(orig_d)
        if d != orig_d:
            x = np.concatenate([x, np.zeros(d - orig_d)])
        signs = WHTransform.random_signs(d, seed)
        x = x * signs
        WHTransform.wht(x)
        return x

    @staticmethod
    def unrotate(x: np.ndarray, orig_dim: int, seed: int = 42) -> np.ndarray:
        """Inverse of rotate(): WHT then undo sign flips, then strip padding.

        Parameters
        ----------
        x : np.ndarray
            Rotated vector (power-of-2 length).
        orig_dim : int
            Original dimension before padding.
        seed : int
            Same seed used during rotate().

        Returns
        -------
        np.ndarray
            Reconstructed vector of length orig_dim.

        Examples
        --------
        >>> v = np.array([1.0, 2.0, 3.0])
        >>> r = WHTransform.rotate(v, seed=7)
        >>> u = WHTransform.unrotate(r, orig_dim=3, seed=7)
        >>> np.allclose(v, u)
        True
        """
        d = len(x)
        WHTransform.wht(x)
        signs = WHTransform.random_signs(d, seed)
        x = x * signs
        return x[:orig_dim]


# ---------------------------------------------------------------------------
# Lloyd-Max Scalar Quantization
# ---------------------------------------------------------------------------

# Pre-computed Lloyd-Max centroids for Gaussian N(0,1) data.
# These are the optimal reconstruction levels that minimise MSE.
# Sources: Jayant & Noll (1984), Max (1960).
_LLOYD_MAX_CENTROIDS: dict[int, np.ndarray | None] = {
    1: np.array([-0.7979, 0.7979]),
    2: np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
    3: np.array([-2.1520, -1.3440, -0.7560, -0.2451,
                  0.2451,  0.7560,  1.3440,  2.1520]),
    4: np.array([-2.7326, -2.0690, -1.6180, -1.2562,
                 -0.9424, -0.6568, -0.3881, -0.1284,
                  0.1284,  0.3881,  0.6568,  0.9424,
                  1.2562,  1.6180,  2.0690,  2.7326]),
    5: np.array([-3.2607, -2.6942, -2.3331, -2.0477, -1.8006,
                 -1.5770, -1.3684, -1.1700, -0.9789, -0.7931,
                 -0.6111, -0.4316, -0.2535, -0.0762,
                  0.0762,  0.2535,  0.4316,  0.6111,
                  0.7931,  0.9789,  1.1700,  1.3684,
                  1.5770,  1.8006,  2.0477,  2.3331,
                  2.6942,  3.2607]),
    6: np.array([
        -3.7451, -3.2472, -2.9355, -2.6928, -2.4843, -2.2969,
        -2.1236, -1.9604, -1.8050, -1.6555, -1.5107, -1.3698,
        -1.2320, -1.0968, -0.9639, -0.8329, -0.7035, -0.5755,
        -0.4487, -0.3229, -0.1980, -0.0739,
         0.0739,  0.1980,  0.3229,  0.4487,  0.5755,  0.7035,
         0.8329,  0.9639,  1.0968,  1.2320,  1.3698,  1.5107,
         1.6555,  1.8050,  1.9604,  2.1236,  2.2969,  2.4843,
         2.6928,  2.9355,  3.2472,  3.7451]),
    7: None,  # generated on demand
    8: None,  # generated on demand
}


def _generate_lloyd_max(n_bits: int, n_iter: int = 200) -> np.ndarray:
    """Run Lloyd-Max optimisation for n_bits on N(0,1) samples.

    Parameters
    ----------
    n_bits : int
        Number of quantization bits (produces 2**n_bits centroids).
    n_iter : int
        Number of EM iterations.

    Returns
    -------
    np.ndarray
        Sorted array of 2**n_bits optimal centroids.
    """
    n_levels = 1 << n_bits
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(100_000)

    # Init centroids uniformly over [-4, 4]
    centroids = np.linspace(-4.0, 4.0, n_levels)

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        indices = np.searchsorted(boundaries, samples)
        new_centroids = np.empty_like(centroids)
        for k in range(n_levels):
            mask = indices == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
            else:
                new_centroids[k] = centroids[k]
        centroids = np.sort(new_centroids)

    return centroids


def lloyd_max_centroids(n_bits: int) -> np.ndarray:
    """Return optimal Lloyd-Max centroids for N(0,1) data.

    Pre-computed for 1-6 bits, generated on first call for 7-8 bits.

    Parameters
    ----------
    n_bits : int
        Quantization bit-width (1-8).

    Returns
    -------
    np.ndarray
        Array of 2**n_bits centroid values, sorted ascending.

    Raises
    ------
    ValueError
        If n_bits is outside the supported range [1, 8].

    Examples
    --------
    >>> c = lloyd_max_centroids(3)
    >>> len(c)
    8
    >>> c[0] < 0 < c[-1]
    True
    """
    if not 1 <= n_bits <= 8:
        raise ValueError(f"n_bits must be 1-8, got {n_bits}")

    if _LLOYD_MAX_CENTROIDS.get(n_bits) is None:
        _LLOYD_MAX_CENTROIDS[n_bits] = _generate_lloyd_max(n_bits)

    return _LLOYD_MAX_CENTROIDS[n_bits].copy()


def quantize_scalar(x: np.ndarray, n_bits: int = 3) -> tuple[np.ndarray, float, float]:
    """Quantize a float vector to n_bits per element using Lloyd-Max.

    The input is first normalised to zero mean, unit variance, then each
    element is mapped to the index of the nearest Lloyd-Max centroid.

    Parameters
    ----------
    x : np.ndarray
        1-D float array to quantize.
    n_bits : int
        Bits per element (1-8).

    Returns
    -------
    indices : np.ndarray
        Array of uint8 centroid indices (values in [0, 2**n_bits - 1]).
    scale : float
        Standard deviation of x (used for reconstruction).
    zero : float
        Mean of x (used for reconstruction).

    Examples
    --------
    >>> idx, s, z = quantize_scalar(np.array([0.0, 1.0, -1.0, 0.5]), n_bits=3)
    >>> idx.dtype
    dtype('uint8')
    >>> len(idx)
    4
    """
    centroids = lloyd_max_centroids(n_bits)
    zero = float(np.mean(x))
    scale = float(np.std(x))
    if scale < 1e-12:
        scale = 1.0

    normalised = (x - zero) / scale

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    indices = np.searchsorted(boundaries, normalised).astype(np.uint8)

    return indices, scale, zero


def dequantize_scalar(
    indices: np.ndarray,
    n_bits: int,
    scale: float,
    zero: float,
) -> np.ndarray:
    """Reconstruct a float vector from quantized indices.

    Parameters
    ----------
    indices : np.ndarray
        Centroid indices (uint8).
    n_bits : int
        Bit-width used during quantization.
    scale : float
        Scale factor from quantize_scalar.
    zero : float
        Zero point from quantize_scalar.

    Returns
    -------
    np.ndarray
        Reconstructed float64 vector.

    Examples
    --------
    >>> c = lloyd_max_centroids(3)
    >>> dequantize_scalar(np.array([0, 7], dtype=np.uint8), 3, 1.0, 0.0)
    array([-2.152,  2.152])
    """
    centroids = lloyd_max_centroids(n_bits)
    return centroids[indices] * scale + zero


# ---------------------------------------------------------------------------
# Bit Packing
# ---------------------------------------------------------------------------

def pack_bits(indices: np.ndarray, n_bits: int) -> bytes:
    """Pack an array of quantization indices into a compact byte string.

    Each index is n_bits wide.  Values are concatenated into a bit stream
    and packed into bytes (MSB first).

    Parameters
    ----------
    indices : np.ndarray
        1-D uint8 array of centroid indices.
    n_bits : int
        Bits per index (1-8).

    Returns
    -------
    bytes
        Packed byte string.

    Examples
    --------
    >>> packed = pack_bits(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8), 3)
    >>> len(packed)  # 8 * 3 bits = 24 bits = 3 bytes
    3
    """
    if n_bits == 8:
        return indices.astype(np.uint8).tobytes()

    total_bits = len(indices) * n_bits
    total_bytes = (total_bits + 7) // 8

    bit_array = np.zeros(total_bytes * 8, dtype=np.uint8)
    for i, idx in enumerate(indices):
        offset = i * n_bits
        for b in range(n_bits):
            bit_array[offset + b] = (int(idx) >> (n_bits - 1 - b)) & 1

    packed = np.packbits(bit_array)
    return packed[:total_bytes].tobytes()


def unpack_bits(packed: bytes, n_bits: int, count: int) -> np.ndarray:
    """Unpack a byte string into an array of quantization indices.

    Parameters
    ----------
    packed : bytes
        Packed byte string from pack_bits.
    n_bits : int
        Bits per index.
    count : int
        Number of indices to unpack.

    Returns
    -------
    np.ndarray
        1-D uint8 array of centroid indices.

    Examples
    --------
    >>> orig = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
    >>> packed = pack_bits(orig, 3)
    >>> recovered = unpack_bits(packed, 3, 8)
    >>> np.array_equal(orig, recovered)
    True
    """
    if n_bits == 8:
        return np.frombuffer(packed, dtype=np.uint8)[:count].copy()

    bit_array = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))

    indices = np.zeros(count, dtype=np.uint8)
    for i in range(count):
        offset = i * n_bits
        val = 0
        for b in range(n_bits):
            val = (val << 1) | int(bit_array[offset + b])
        indices[i] = val

    return indices


# ---------------------------------------------------------------------------
# CrowQuantBlock and Main API
# ---------------------------------------------------------------------------

@dataclass
class CrowQuantBlock:
    """Container for a quantized vector.

    Attributes
    ----------
    scale : float
        Std dev of the original (rotated) vector.
    zero : float
        Mean of the original (rotated) vector.
    n_bits : int
        Bits per element.
    dim : int
        Original vector dimension (before any padding).
    seed : int
        RNG seed used for WHT rotation.
    packed_data : bytes
        Bit-packed centroid indices.
    padded_dim : int
        Dimension after power-of-2 padding (needed for unrotation).
    """

    scale: float
    zero: float
    n_bits: int
    dim: int
    seed: int
    packed_data: bytes
    padded_dim: int

    @property
    def compressed_size(self) -> int:
        """Total size in bytes (packed data + metadata overhead).

        Examples
        --------
        >>> blk = quantize(np.random.randn(128), n_bits=3)
        >>> blk.compressed_size > 0
        True
        """
        return len(self.packed_data) + 32

    @property
    def compression_ratio(self) -> float:
        """Ratio of original size (float64) to compressed size.

        Examples
        --------
        >>> blk = quantize(np.random.randn(128), n_bits=3)
        >>> blk.compression_ratio > 1.0
        True
        """
        original = self.dim * 8  # float64 = 8 bytes
        return original / self.compressed_size


def quantize(
    vector: np.ndarray,
    n_bits: int = 3,
    seed: int = 42,
) -> CrowQuantBlock:
    """Quantize a vector: rotate -> scalar quantize -> bit pack.

    Full CrowQuant compression pipeline.  The vector is first rotated
    via randomised Walsh-Hadamard Transform to decorrelate components,
    then each component is quantized to the nearest Lloyd-Max centroid,
    and finally the indices are bit-packed.

    Parameters
    ----------
    vector : np.ndarray
        1-D float vector to compress.
    n_bits : int
        Bits per element (1-8). Default 3 (8x compression from float64).
    seed : int
        RNG seed for reproducible WHT rotation.

    Returns
    -------
    CrowQuantBlock
        Compressed representation.

    Examples
    --------
    >>> v = np.random.randn(256)
    >>> blk = quantize(v, n_bits=3, seed=42)
    >>> blk.dim
    256
    >>> blk.n_bits
    3
    >>> reconstructed = dequantize(blk)
    >>> reconstructed.shape
    (256,)
    """
    vector = np.asarray(vector, dtype=np.float64).ravel()
    orig_dim = len(vector)

    rotated = WHTransform.rotate(vector.copy(), seed=seed)
    padded_dim = len(rotated)

    indices, scale, zero = quantize_scalar(rotated, n_bits)

    packed = pack_bits(indices, n_bits)

    return CrowQuantBlock(
        scale=scale,
        zero=zero,
        n_bits=n_bits,
        dim=orig_dim,
        seed=seed,
        packed_data=packed,
        padded_dim=padded_dim,
    )


def dequantize(block: CrowQuantBlock) -> np.ndarray:
    """Reconstruct a vector from a CrowQuantBlock.

    Inverse of quantize(): unpack -> dequantize -> unrotate.

    Parameters
    ----------
    block : CrowQuantBlock
        Compressed vector block.

    Returns
    -------
    np.ndarray
        Reconstructed float64 vector of length block.dim.

    Examples
    --------
    >>> v = np.random.randn(100)
    >>> blk = quantize(v, n_bits=4)
    >>> r = dequantize(blk)
    >>> r.shape
    (100,)
    >>> np.corrcoef(v, r)[0, 1] > 0.9  # high correlation
    True
    """
    indices = unpack_bits(block.packed_data, block.n_bits, block.padded_dim)

    rotated = dequantize_scalar(indices, block.n_bits, block.scale, block.zero)

    return WHTransform.unrotate(rotated, block.dim, block.seed)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_block(block: CrowQuantBlock) -> bytes:
    """Serialize a CrowQuantBlock to bytes for storage.

    Format: header (scale f64, zero f64, n_bits u8, dim u32, seed u32,
    padded_dim u32, data_len u32) followed by packed_data bytes.

    Parameters
    ----------
    block : CrowQuantBlock
        Block to serialize.

    Returns
    -------
    bytes
        Serialized representation.
    """
    import struct
    header = struct.pack(
        '<ddBIIII',
        block.scale,
        block.zero,
        block.n_bits,
        block.dim,
        block.seed,
        block.padded_dim,
        len(block.packed_data),
    )
    return header + block.packed_data


def deserialize_block(data: bytes) -> CrowQuantBlock:
    """Deserialize bytes back into a CrowQuantBlock.

    Parameters
    ----------
    data : bytes
        Output of serialize_block.

    Returns
    -------
    CrowQuantBlock
        Reconstructed block.
    """
    import struct
    header_size = struct.calcsize('<ddBIIII')
    scale, zero, n_bits, dim, seed, padded_dim, data_len = struct.unpack(
        '<ddBIIII', data[:header_size]
    )
    packed_data = data[header_size:header_size + data_len]
    return CrowQuantBlock(
        scale=scale,
        zero=zero,
        n_bits=n_bits,
        dim=dim,
        seed=seed,
        packed_data=packed_data,
        padded_dim=padded_dim,
    )
