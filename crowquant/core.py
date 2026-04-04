"""CrowQuant core -- WHT-based adaptive vector quantization."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import struct


@dataclass
class CrowQuantBlock:
    """A compressed vector block."""
    n_bits: int
    dim: int
    scale: float
    zero_point: float
    packed: bytes  # bit-packed quantized indices
    outlier_mask: Optional[bytes] = None  # bitmask for outlier channels
    outlier_values: Optional[bytes] = None  # float16 outlier values


class WHTransform:
    """Walsh-Hadamard Transform for decorrelating embedding dimensions.

    WHT is an orthogonal transform (like DCT/FFT) but uses only +1/-1,
    making it extremely fast. It decorrelates typical embedding dimensions,
    concentrating energy into fewer coefficients for better quantization.
    """

    @staticmethod
    def _hadamard_matrix(n: int) -> np.ndarray:
        """Build Hadamard matrix of size n (must be power of 2)."""
        if n == 1:
            return np.array([[1.0]])
        half = WHTransform._hadamard_matrix(n // 2)
        return np.block([[half, half], [half, -half]]) / np.sqrt(2)

    @staticmethod
    def pad_to_pow2(x: np.ndarray) -> tuple[np.ndarray, int]:
        """Pad vector to next power of 2."""
        orig_len = len(x)
        n = 1
        while n < orig_len:
            n <<= 1
        if n == orig_len:
            return x, orig_len
        padded = np.zeros(n, dtype=x.dtype)
        padded[:orig_len] = x
        return padded, orig_len

    @staticmethod
    def fast_wht(x: np.ndarray) -> np.ndarray:
        """In-place iterative fast Walsh-Hadamard transform.

        O(n log n) using butterfly operations, normalized by 1/sqrt(n).
        """
        x = x.astype(np.float64).copy()
        n = len(x)
        assert n > 0 and (n & (n - 1)) == 0, f"length must be power of 2, got {n}"

        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    a = x[j]
                    b = x[j + h]
                    x[j] = a + b
                    x[j + h] = a - b
            h *= 2

        x /= np.sqrt(n)
        return x

    @staticmethod
    def fast_iwht(x: np.ndarray) -> np.ndarray:
        """Inverse fast WHT. For normalized WHT, inverse = forward."""
        return WHTransform.fast_wht(x)

    @staticmethod
    def forward(x: np.ndarray) -> tuple[np.ndarray, int]:
        """Forward WHT with automatic padding."""
        padded, orig_len = WHTransform.pad_to_pow2(x)
        transformed = WHTransform.fast_wht(padded)
        return transformed, orig_len

    @staticmethod
    def inverse(x: np.ndarray, orig_len: int) -> np.ndarray:
        """Inverse WHT with unpadding."""
        recovered = WHTransform.fast_iwht(x)
        return recovered[:orig_len]


def pack_bits(indices: np.ndarray, n_bits: int) -> bytes:
    """Pack an array of n-bit unsigned integers into bytes.

    Packs indices (each in [0, 2^n_bits - 1]) into a compact byte string
    using little-endian bit ordering within each byte.
    """
    indices = indices.astype(np.uint64)
    total_bits = len(indices) * n_bits
    n_bytes = (total_bits + 7) // 8
    result = np.zeros(n_bytes, dtype=np.uint8)

    bit_pos = 0
    for idx in indices:
        for b in range(n_bits):
            if idx & (1 << b):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                result[byte_idx] |= (1 << bit_idx)
            bit_pos += 1

    return result.tobytes()


def unpack_bits(data: bytes, n_bits: int, count: int) -> np.ndarray:
    """Unpack a byte string into an array of n-bit unsigned integers."""
    buf = np.frombuffer(data, dtype=np.uint8)
    result = np.zeros(count, dtype=np.uint64)

    bit_pos = 0
    for i in range(count):
        val = 0
        for b in range(n_bits):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            if buf[byte_idx] & (1 << bit_idx):
                val |= (1 << b)
            bit_pos += 1
        result[i] = val

    return result


def quantize(x: np.ndarray, n_bits: int = 4, use_wht: bool = True) -> CrowQuantBlock:
    """Quantize a float32 vector to n-bit representation.

    Args:
        x: Input vector (float32 or float64).
        n_bits: Bits per dimension (1-8). Default 4.
        use_wht: Apply Walsh-Hadamard Transform before quantizing.

    Returns:
        CrowQuantBlock with compressed representation.
    """
    x = np.asarray(x, dtype=np.float64)
    dim = len(x)
    orig_len = dim

    if use_wht:
        coeffs, orig_len = WHTransform.forward(x)
    else:
        coeffs = x.copy()

    n_levels = (1 << n_bits)
    vmin = coeffs.min()
    vmax = coeffs.max()

    # avoid division by zero for constant vectors
    if vmax == vmin:
        scale = 1.0
        zero_point = vmin
        indices = np.zeros(len(coeffs), dtype=np.uint64)
    else:
        scale = (vmax - vmin) / (n_levels - 1)
        zero_point = vmin
        indices = np.round((coeffs - zero_point) / scale).astype(np.uint64)
        indices = np.clip(indices, 0, n_levels - 1)

    packed = pack_bits(indices, n_bits)

    return CrowQuantBlock(
        n_bits=n_bits,
        dim=dim,
        scale=scale,
        zero_point=zero_point,
        packed=packed,
    )


def dequantize(block: CrowQuantBlock, use_wht: bool = True) -> np.ndarray:
    """Reconstruct a float vector from a CrowQuantBlock.

    Args:
        block: Compressed block.
        use_wht: Whether WHT was used during quantization.

    Returns:
        Reconstructed float64 vector of length block.dim.
    """
    padded_len = len(block.packed) * 8 // block.n_bits
    # actual count of packed values is the padded dim (power of 2 if WHT used)
    if use_wht:
        # figure out padded length
        n = 1
        while n < block.dim:
            n <<= 1
        count = n
    else:
        count = block.dim

    indices = unpack_bits(block.packed, block.n_bits, count)
    coeffs = indices.astype(np.float64) * block.scale + block.zero_point

    if use_wht:
        return WHTransform.inverse(coeffs, block.dim)
    else:
        return coeffs[:block.dim]


def serialize_block(block: CrowQuantBlock) -> bytes:
    """Serialize a CrowQuantBlock to bytes for storage."""
    # header: n_bits(1B) + dim(4B) + scale(8B) + zero_point(8B) + packed_len(4B)
    header = struct.pack('<BIddi', block.n_bits, block.dim, block.scale,
                         block.zero_point, len(block.packed))
    has_outliers = block.outlier_mask is not None
    flags = struct.pack('<B', 1 if has_outliers else 0)
    parts = [header, flags, block.packed]
    if has_outliers:
        parts.append(struct.pack('<I', len(block.outlier_mask)))
        parts.append(block.outlier_mask)
        parts.append(struct.pack('<I', len(block.outlier_values)))
        parts.append(block.outlier_values)
    return b''.join(parts)


def deserialize_block(data: bytes) -> CrowQuantBlock:
    """Deserialize bytes back into a CrowQuantBlock."""
    # header: 1 + 4 + 8 + 8 + 4 = 25 bytes
    offset = 0
    n_bits, dim, scale, zero_point, packed_len = struct.unpack_from('<BIddi', data, offset)
    offset += struct.calcsize('<BIddi')
    flags = struct.unpack_from('<B', data, offset)[0]
    offset += 1
    packed = data[offset:offset + packed_len]
    offset += packed_len

    outlier_mask = None
    outlier_values = None
    if flags & 1:
        mask_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        outlier_mask = data[offset:offset + mask_len]
        offset += mask_len
        ov_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        outlier_values = data[offset:offset + ov_len]
        offset += ov_len

    return CrowQuantBlock(
        n_bits=n_bits,
        dim=dim,
        scale=scale,
        zero_point=zero_point,
        packed=packed,
        outlier_mask=outlier_mask,
        outlier_values=outlier_values,
    )
