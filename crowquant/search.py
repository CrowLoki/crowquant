"""Compressed-domain similarity search."""
import numpy as np
from typing import Optional
from .core import CrowQuantBlock, unpack_bits, dequantize, quantize


def compressed_dot_product(block_a: CrowQuantBlock, block_b: CrowQuantBlock,
                           use_wht: bool = True) -> float:
    """Compute approximate dot product between two compressed vectors.

    Dequantizes both vectors and computes their dot product. While this
    doesn't save compute on the dot product itself, it enables storage
    of vectors in compressed form with on-the-fly decompression.
    """
    a = dequantize(block_a, use_wht=use_wht)
    b = dequantize(block_b, use_wht=use_wht)
    return float(np.dot(a, b))


def compressed_cosine(block_a: CrowQuantBlock, block_b: CrowQuantBlock,
                      use_wht: bool = True) -> float:
    """Compute approximate cosine similarity between two compressed vectors."""
    a = dequantize(block_a, use_wht=use_wht)
    b = dequantize(block_b, use_wht=use_wht)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compressed_knn(query: np.ndarray, blocks: list[CrowQuantBlock],
                   k: int = 5, metric: str = "cosine",
                   use_wht: bool = True) -> list[tuple[int, float]]:
    """Find k nearest neighbors from a list of compressed blocks.

    Args:
        query: Uncompressed query vector.
        blocks: List of CrowQuantBlocks to search.
        k: Number of neighbors to return.
        metric: "cosine" or "dot".
        use_wht: Whether WHT was used during compression.

    Returns:
        List of (index, score) tuples, sorted by descending score.
    """
    scores = []
    for i, block in enumerate(blocks):
        vec = dequantize(block, use_wht=use_wht)
        if metric == "cosine":
            norm_q = np.linalg.norm(query)
            norm_v = np.linalg.norm(vec)
            if norm_q < 1e-10 or norm_v < 1e-10:
                score = 0.0
            else:
                score = float(np.dot(query, vec) / (norm_q * norm_v))
        elif metric == "dot":
            score = float(np.dot(query, vec))
        else:
            raise ValueError(f"unknown metric: {metric}")
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def batch_dequantize(blocks: list[CrowQuantBlock],
                     use_wht: bool = True) -> np.ndarray:
    """Dequantize a batch of blocks into a 2D array."""
    if not blocks:
        return np.array([], dtype=np.float64)
    vecs = [dequantize(b, use_wht=use_wht) for b in blocks]
    return np.stack(vecs)
