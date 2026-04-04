"""Compressed vector search operations for CrowQuant.

The key innovation: dot products and KNN directly on compressed
representations, avoiding full decompression.  Instead of reconstructing
float vectors, we use centroid lookup tables to approximate inner
products in O(d) with tiny constants.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import CrowQuantBlock, lloyd_max_centroids, unpack_bits


def _get_indices(block: CrowQuantBlock) -> np.ndarray:
    """Unpack centroid indices from a CrowQuantBlock.

    Parameters
    ----------
    block : CrowQuantBlock
        Compressed vector.

    Returns
    -------
    np.ndarray
        uint8 array of centroid indices, length = block.padded_dim.
    """
    return unpack_bits(block.packed_data, block.n_bits, block.padded_dim)


def compressed_dot_product(block_a: CrowQuantBlock, block_b: CrowQuantBlock) -> float:
    """Approximate dot product between two CrowQuant blocks WITHOUT full decompression.

    Uses centroid-based computation: instead of reconstructing full float
    vectors, we look up centroid values for each index pair and accumulate
    the product.  This is O(d) with small constants -- just index lookups
    and multiplies, no WHT inverse needed.

    For best accuracy both blocks should use the same n_bits and seed,
    but the function handles mismatched settings gracefully.

    Parameters
    ----------
    block_a : CrowQuantBlock
        First compressed vector.
    block_b : CrowQuantBlock
        Second compressed vector.

    Returns
    -------
    float
        Approximate dot product <a, b>.

    Notes
    -----
    The approximation is exact in the quantized domain: it computes the
    true dot product of the dequantized (but still rotated) vectors.
    Since WHT is orthogonal, the dot product is preserved under rotation,
    so this equals the dot product of the dequantized output vectors.

    Examples
    --------
    >>> from crowquant.core import quantize, dequantize
    >>> import numpy as np
    >>> a = np.random.randn(128)
    >>> b = np.random.randn(128)
    >>> ba = quantize(a, n_bits=4)
    >>> bb = quantize(b, n_bits=4)
    >>> approx = compressed_dot_product(ba, bb)
    >>> exact = np.dot(dequantize(ba), dequantize(bb))
    >>> abs(approx - exact) < 1e-6
    True
    """
    centroids_a = lloyd_max_centroids(block_a.n_bits)
    centroids_b = lloyd_max_centroids(block_b.n_bits)

    idx_a = _get_indices(block_a)
    idx_b = _get_indices(block_b)

    n = min(len(idx_a), len(idx_b))

    vals_a = centroids_a[idx_a[:n]] * block_a.scale + block_a.zero
    vals_b = centroids_b[idx_b[:n]] * block_b.scale + block_b.zero

    return float(np.dot(vals_a, vals_b))


def compressed_cosine(block_a: CrowQuantBlock, block_b: CrowQuantBlock) -> float:
    """Approximate cosine similarity from compressed blocks.

    Computes cos(a, b) = dot(a, b) / (||a|| * ||b||) using centroid
    lookups -- no full decompression needed.

    Parameters
    ----------
    block_a : CrowQuantBlock
        First compressed vector.
    block_b : CrowQuantBlock
        Second compressed vector.

    Returns
    -------
    float
        Approximate cosine similarity in [-1, 1].

    Examples
    --------
    >>> from crowquant.core import quantize
    >>> import numpy as np
    >>> v = np.random.randn(64)
    >>> ba = quantize(v, n_bits=4)
    >>> bb = quantize(v, n_bits=4)
    >>> compressed_cosine(ba, bb) > 0.99
    True
    """
    centroids_a = lloyd_max_centroids(block_a.n_bits)
    centroids_b = lloyd_max_centroids(block_b.n_bits)

    idx_a = _get_indices(block_a)
    idx_b = _get_indices(block_b)

    n = min(len(idx_a), len(idx_b))

    vals_a = centroids_a[idx_a[:n]] * block_a.scale + block_a.zero
    vals_b = centroids_b[idx_b[:n]] * block_b.scale + block_b.zero

    dot_ab = np.dot(vals_a, vals_b)
    norm_a = np.linalg.norm(vals_a)
    norm_b = np.linalg.norm(vals_b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(dot_ab / (norm_a * norm_b))


def compressed_knn(
    query_block: CrowQuantBlock,
    database_blocks: Sequence[CrowQuantBlock],
    k: int = 5,
) -> list[tuple[int, float]]:
    """Find k nearest neighbors using compressed dot products.

    Scores each database block against the query using compressed_dot_product
    and returns the top-k by score (highest dot product = most similar,
    assuming normalised embeddings).

    Parameters
    ----------
    query_block : CrowQuantBlock
        The query vector (compressed).
    database_blocks : sequence of CrowQuantBlock
        Database of compressed vectors to search.
    k : int
        Number of nearest neighbors to return.

    Returns
    -------
    list of (index, score) tuples
        Top-k results sorted by descending dot product score.

    Examples
    --------
    >>> from crowquant.core import quantize
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> db = [quantize(rng.standard_normal(64), n_bits=3) for _ in range(20)]
    >>> query = db[5]
    >>> results = compressed_knn(query, db, k=3)
    >>> results[0][0]  # top hit should be index 5 (itself)
    5
    >>> len(results)
    3
    """
    k = min(k, len(database_blocks))

    scores = []
    for i, db_block in enumerate(database_blocks):
        score = compressed_dot_product(query_block, db_block)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
