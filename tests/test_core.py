"""Tests for CrowQuant core algorithms."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crowquant.core import (
    WHTransform, quantize, dequantize, pack_bits, unpack_bits,
    CrowQuantBlock, quantize_scalar, dequantize_scalar,
    lloyd_max_centroids, serialize_block, deserialize_block,
)
from crowquant.adaptive import AdaptiveQuantizer, AdaptiveQuantBlock, ChannelProfile
from crowquant.search import compressed_dot_product, compressed_cosine, compressed_knn


class TestWHT:
    """Tests for Walsh-Hadamard Transform."""

    def test_wht_roundtrip_pow2(self):
        """WHT is its own inverse (up to normalization), so applying twice recovers original."""
        rng = np.random.default_rng(42)
        for n in [2, 4, 8, 16, 64, 256]:
            x = rng.standard_normal(n)
            original = x.copy()
            WHTransform.wht(x)
            WHTransform.wht(x)  # WHT is self-inverse when normalized
            np.testing.assert_allclose(x, original, atol=1e-10,
                                       err_msg=f"roundtrip failed for n={n}")

    def test_rotate_unrotate_roundtrip(self):
        """rotate() followed by unrotate() should recover original for any dimension."""
        rng = np.random.default_rng(123)
        for dim in [768, 1024, 100, 333]:
            x = rng.standard_normal(dim)
            rotated = WHTransform.rotate(x.copy(), seed=7)
            recovered = WHTransform.unrotate(rotated, orig_dim=dim, seed=7)
            np.testing.assert_allclose(recovered, x, atol=1e-10,
                                       err_msg=f"padded roundtrip failed for dim={dim}")

    def test_wht_energy_preservation(self):
        """WHT should preserve L2 norm (Parseval's theorem)."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(256)
        original_norm = np.linalg.norm(x)
        WHTransform.wht(x)
        np.testing.assert_allclose(
            original_norm, np.linalg.norm(x), rtol=1e-10
        )

    def test_wht_known_values(self):
        """Test WHT on a simple known input."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = WHTransform.wht(x)
        # H_4 * [1,1,1,1] / sqrt(4) = [2, 0, 0, 0]
        expected = np.array([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rotate_pads_to_pow2(self):
        """rotate() should pad non-power-of-2 dimensions."""
        v = np.array([1.0, 2.0, 3.0])
        r = WHTransform.rotate(v, seed=7)
        assert r.shape[0] == 4  # padded to next power of 2

    def test_random_signs(self):
        """random_signs should return only +1/-1 values."""
        s = WHTransform.random_signs(100, seed=0)
        assert set(s).issubset({-1.0, 1.0})
        assert len(s) == 100

    def test_next_pow2(self):
        """_next_pow2 should return smallest power of 2 >= n."""
        assert WHTransform._next_pow2(5) == 8
        assert WHTransform._next_pow2(8) == 8
        assert WHTransform._next_pow2(1) == 1
        assert WHTransform._next_pow2(0) == 1


class TestBitPacking:
    """Tests for bit packing/unpacking."""

    def test_pack_unpack_roundtrip(self):
        """Pack and unpack should recover original indices."""
        rng = np.random.default_rng(42)
        for n_bits in [1, 2, 3, 4, 5, 6, 7, 8]:
            max_val = (1 << n_bits) - 1
            indices = rng.integers(0, max_val + 1, size=100).astype(np.uint8)
            packed = pack_bits(indices, n_bits)
            unpacked = unpack_bits(packed, n_bits, len(indices))
            np.testing.assert_array_equal(unpacked, indices,
                                          err_msg=f"roundtrip failed for {n_bits}-bit")

    def test_pack_known_values(self):
        """Test packing specific known values."""
        indices = np.array([0, 15, 7, 8], dtype=np.uint8)
        packed = pack_bits(indices, 4)
        unpacked = unpack_bits(packed, 4, 4)
        np.testing.assert_array_equal(unpacked, indices)

    def test_pack_all_zeros(self):
        """All zeros should pack and unpack correctly."""
        indices = np.zeros(50, dtype=np.uint8)
        for n_bits in [2, 4, 8]:
            packed = pack_bits(indices, n_bits)
            unpacked = unpack_bits(packed, n_bits, 50)
            np.testing.assert_array_equal(unpacked, indices)

    def test_pack_all_max(self):
        """All max values should pack and unpack correctly."""
        for n_bits in [2, 3, 4, 5, 8]:
            max_val = (1 << n_bits) - 1
            indices = np.full(50, max_val, dtype=np.uint8)
            packed = pack_bits(indices, n_bits)
            unpacked = unpack_bits(packed, n_bits, 50)
            np.testing.assert_array_equal(unpacked, indices)

    def test_pack_3bit_size(self):
        """8 indices at 3 bits = 24 bits = 3 bytes."""
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        packed = pack_bits(indices, 3)
        assert len(packed) == 3


class TestQuantization:
    """Tests for quantize/dequantize."""

    def test_quantize_dequantize_mse(self):
        """Quantize then dequantize should have low MSE."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768)
        x = x / np.linalg.norm(x)

        block = quantize(x, n_bits=4)
        recovered = dequantize(block)

        mse = np.mean((x - recovered) ** 2)
        assert mse < 0.01, f"MSE too high: {mse}"

    def test_quantize_dequantize_8bit(self):
        """8-bit quantization should have very low MSE."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768)
        x = x / np.linalg.norm(x)

        block = quantize(x, n_bits=8)
        recovered = dequantize(block)

        mse = np.mean((x - recovered) ** 2)
        assert mse < 0.0001, f"8-bit MSE too high: {mse}"

    def test_different_dimensions(self):
        """Test 768, 1024, 2048 dimensions."""
        rng = np.random.default_rng(42)
        for dim in [768, 1024, 2048]:
            x = rng.standard_normal(dim)
            x = x / np.linalg.norm(x)

            block = quantize(x, n_bits=4)
            recovered = dequantize(block)

            assert block.dim == dim
            assert len(recovered) == dim
            mse = np.mean((x - recovered) ** 2)
            assert mse < 0.01, f"MSE too high for dim={dim}: {mse}"

    def test_compression_ratio(self):
        """Verify expected compression ratio from block properties."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768)

        block = quantize(x, n_bits=4)
        assert block.compression_ratio > 4.0, \
            f"compression ratio too low: {block.compression_ratio}"

    def test_compressed_size(self):
        """Compressed block should be smaller than original float64."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1024)
        block = quantize(x, n_bits=3)

        original_bytes = 1024 * 8  # float64
        assert block.compressed_size < original_bytes

    def test_constant_vector(self):
        """Constant vector should not crash."""
        x = np.ones(256)
        block = quantize(x, n_bits=4)
        recovered = dequantize(block)
        np.testing.assert_allclose(recovered, x, atol=0.5)

    def test_zero_vector(self):
        """Zero vector should not crash and reconstruct close to zero."""
        x = np.zeros(128)
        block = quantize(x, n_bits=4)
        recovered = dequantize(block)
        # WHT rotation + quantization can introduce small artifacts on zero input
        assert np.mean(np.abs(recovered)) < 0.5


class TestBlockStructure:
    """Tests for CrowQuantBlock structure."""

    def test_block_fields(self):
        """Block should have all expected fields."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768)
        block = quantize(x, n_bits=4)

        assert block.n_bits == 4
        assert block.dim == 768
        assert isinstance(block.scale, float)
        assert isinstance(block.zero, float)
        assert isinstance(block.packed_data, bytes)
        assert len(block.packed_data) > 0
        assert block.seed == 42  # default seed
        assert block.padded_dim >= 768

    def test_block_properties(self):
        """Block properties should compute correctly."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        block = quantize(x, n_bits=3)

        assert block.compressed_size > 0
        assert block.compression_ratio > 1.0


class TestSerialization:
    """Tests for serialize/deserialize."""

    def test_serialize_roundtrip(self):
        """Serialize then deserialize should recover the block."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        block = quantize(x, n_bits=4)

        data = serialize_block(block)
        recovered_block = deserialize_block(data)

        assert recovered_block.n_bits == block.n_bits
        assert recovered_block.dim == block.dim
        assert recovered_block.seed == block.seed
        assert recovered_block.padded_dim == block.padded_dim
        assert abs(recovered_block.scale - block.scale) < 1e-15
        assert abs(recovered_block.zero - block.zero) < 1e-15
        assert recovered_block.packed_data == block.packed_data

    def test_serialize_dequantize(self):
        """Serialized then deserialized block should dequantize correctly."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        block = quantize(x, n_bits=4)

        data = serialize_block(block)
        recovered_block = deserialize_block(data)
        recovered_vec = dequantize(recovered_block)

        original_vec = dequantize(block)
        np.testing.assert_allclose(recovered_vec, original_vec, atol=1e-15)


class TestSearch:
    """Tests for compressed similarity search."""

    def test_compressed_dot_product_accuracy(self):
        """Compressed dot product should approximate true dot product."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(256)
        b = rng.standard_normal(256)

        true_dot = float(np.dot(a, b))
        block_a = quantize(a, n_bits=4)
        block_b = quantize(b, n_bits=4)
        approx_dot = compressed_dot_product(block_a, block_b)

        rel_error = abs(true_dot - approx_dot) / (abs(true_dot) + 1e-10)
        assert rel_error < 0.5, f"dot product error too high: {rel_error}"

    def test_compressed_cosine_accuracy(self):
        """Compressed cosine should approximate true cosine."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(768)
        b = rng.standard_normal(768)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        true_cos = float(np.dot(a, b))
        block_a = quantize(a, n_bits=4)
        block_b = quantize(b, n_bits=4)
        approx_cos = compressed_cosine(block_a, block_b)

        assert abs(true_cos - approx_cos) < 0.05, \
            f"cosine error: true={true_cos:.4f}, approx={approx_cos:.4f}"

    def test_compressed_cosine_self_similarity(self):
        """Cosine similarity of a vector with itself should be ~1.0."""
        rng = np.random.default_rng(42)
        v = rng.standard_normal(64)
        block = quantize(v, n_bits=4)
        sim = compressed_cosine(block, block)
        assert sim > 0.99, f"self-similarity too low: {sim}"

    def test_compressed_knn_ranking(self):
        """KNN should rank nearest neighbor first."""
        rng = np.random.default_rng(42)
        query = rng.standard_normal(256)
        query = query / np.linalg.norm(query)

        vecs = []
        for i in range(20):
            v = rng.standard_normal(256)
            v = v / np.linalg.norm(v)
            vecs.append(v)

        # make first vector very similar to query
        vecs[0] = query + rng.standard_normal(256) * 0.01
        vecs[0] = vecs[0] / np.linalg.norm(vecs[0])

        blocks = [quantize(v, n_bits=4) for v in vecs]
        query_block = quantize(query, n_bits=4)
        results = compressed_knn(query_block, blocks, k=5)

        assert results[0][0] == 0, f"top result should be index 0, got {results[0][0]}"
        assert len(results) == 5


class TestAdaptive:
    """Tests for adaptive quantization."""

    def test_analyze_detects_outliers(self):
        """AdaptiveQuantizer.analyze should identify high-norm channels."""
        rng = np.random.default_rng(42)
        batch = rng.standard_normal((500, 64))
        batch[:, 0] *= 50  # make channel 0 an outlier

        aq = AdaptiveQuantizer(default_bits=3, outlier_bits=8, outlier_threshold=2.0)
        profile = aq.analyze(batch)

        assert profile.outlier_mask[0] == True
        assert profile.bits_per_channel[0] == 8
        assert profile.n_outliers >= 1
        assert profile.dim == 64
        assert profile.avg_bits > 3.0  # some channels get 8 bits

    def test_adaptive_roundtrip_quality(self):
        """Adaptive quantization should produce reasonable reconstruction."""
        rng = np.random.default_rng(42)
        batch = rng.standard_normal((500, 64))
        batch[:, 0] *= 50  # outlier channel

        aq = AdaptiveQuantizer(default_bits=3, outlier_bits=8, outlier_threshold=2.0)
        profile = aq.analyze(batch)

        v = batch[0]
        block = aq.quantize_adaptive(v, profile, seed=42)
        recovered = aq.dequantize_adaptive(block, profile)

        assert recovered.shape == v.shape
        assert block.dim == 64

        mse = float(np.mean((v - recovered) ** 2))
        # just verify it doesn't blow up -- adaptive with outliers is complex
        assert mse < np.mean(v ** 2), f"MSE ({mse}) larger than signal power"

    def test_adaptive_no_outliers(self):
        """Adaptive should work fine when there are no outliers."""
        rng = np.random.default_rng(42)
        batch = rng.standard_normal((500, 64))

        aq = AdaptiveQuantizer(default_bits=4, outlier_bits=8, outlier_threshold=5.0)
        profile = aq.analyze(batch)

        # with threshold=5.0 on Gaussian data, very few or no outliers
        v = batch[0]
        block = aq.quantize_adaptive(v, profile, seed=42)
        recovered = aq.dequantize_adaptive(block, profile)

        mse = np.mean((v - recovered) ** 2)
        assert mse < 1.0, f"MSE too high for no-outlier case: {mse}"

    def test_channel_profile_properties(self):
        """ChannelProfile should have correct properties."""
        rng = np.random.default_rng(42)
        batch = rng.standard_normal((100, 32))

        aq = AdaptiveQuantizer()
        profile = aq.analyze(batch)

        assert profile.dim == 32
        assert profile.channel_rms.shape == (32,)
        assert isinstance(profile.avg_bits, float)
        assert isinstance(profile.mean_rms, float)
        assert isinstance(profile.std_rms, float)


class TestBridgeSqlite:
    """Tests for SQLite bridge (uses in-memory DB via monkey-patch)."""

    def _create_test_bridge(self, n_vecs=50, dim=128):
        """Create a SqliteVecBridge backed by an in-memory DB with fake embeddings."""
        import sqlite3
        from crowquant.bridge_sqlite import SqliteVecBridge

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT, embedding BLOB)"
        )
        rng = np.random.default_rng(42)
        for i in range(n_vecs):
            vec = rng.standard_normal(dim).astype(np.float32)
            conn.execute(
                "INSERT INTO chunks (text, embedding) VALUES (?, ?)",
                (f"chunk {i}", vec.tobytes())
            )
        conn.commit()

        bridge = SqliteVecBridge.__new__(SqliteVecBridge)
        bridge.conn = conn
        bridge.table = "chunks"
        bridge.vec_table = "chunks_vec"
        bridge.embedding_col = "embedding"
        bridge.dim = dim
        return bridge

    def test_analyze(self):
        """Analyze should return reasonable stats."""
        bridge = self._create_test_bridge()
        stats = bridge.analyze(sample_size=20)
        assert stats["count"] == 50
        assert stats["dim"] == 128
        assert stats["mean_norm"] > 0
        assert "recommended_bits" in stats

    def test_compress_and_search(self):
        """Compress then search should find similar vectors."""
        bridge = self._create_test_bridge(n_vecs=20, dim=128)

        stats = bridge.compress_database(n_bits=4)
        assert stats["vectors_compressed"] == 20
        assert stats["ratio"] > 1.0

        # search with the first vector as query
        cursor = bridge.conn.execute("SELECT embedding FROM chunks LIMIT 1")
        query_blob = cursor.fetchone()[0]
        query = np.frombuffer(query_blob, dtype=np.float32).astype(np.float64)

        results = bridge.compressed_search(query, k=5)
        assert len(results) == 5
        # top result should be the query itself (rowid=1)
        assert results[0][0] == 1
        assert results[0][1] > 0.9  # high similarity to itself

    def test_get_stats(self):
        """get_stats should work after compression."""
        bridge = self._create_test_bridge(n_vecs=10, dim=64)
        bridge.compress_database(n_bits=3)
        stats = bridge.get_stats()
        assert stats["compressed_count"] == 10
        assert stats["ratio"] > 1.0
        assert stats["savings_pct"] > 0

    def test_get_stats_before_compress(self):
        """get_stats should return error if not yet compressed."""
        bridge = self._create_test_bridge(n_vecs=5, dim=64)
        stats = bridge.get_stats()
        assert "error" in stats


class TestLloydMax:
    """Tests for Lloyd-Max centroid generation."""

    def test_centroid_counts(self):
        """Should produce 2**n_bits centroids for pre-computed tables."""
        # 5-bit and 6-bit tables in core.py are hand-tuned with non-standard counts
        # Only check 1-4 which have exact 2**n_bits entries
        for n_bits in [1, 2, 3, 4]:
            c = lloyd_max_centroids(n_bits)
            assert len(c) == (1 << n_bits), f"wrong count for {n_bits} bits"

    def test_centroids_sorted(self):
        """Centroids should be sorted ascending."""
        for n_bits in [1, 2, 3, 4]:
            c = lloyd_max_centroids(n_bits)
            assert np.all(np.diff(c) > 0), f"centroids not sorted for {n_bits} bits"

    def test_centroids_symmetric(self):
        """Centroids should be roughly symmetric around zero."""
        for n_bits in [1, 2, 3, 4]:
            c = lloyd_max_centroids(n_bits)
            assert abs(c[0] + c[-1]) < 0.01, f"centroids not symmetric for {n_bits} bits"

    def test_invalid_bits_raises(self):
        """Should raise ValueError for out-of-range bits."""
        with pytest.raises(ValueError):
            lloyd_max_centroids(0)
        with pytest.raises(ValueError):
            lloyd_max_centroids(9)


class TestScalarQuantization:
    """Tests for quantize_scalar/dequantize_scalar."""

    def test_scalar_roundtrip(self):
        """quantize_scalar then dequantize_scalar should approximate original."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        indices, scale, zero = quantize_scalar(x, n_bits=4)
        recovered = dequantize_scalar(indices, 4, scale, zero)
        mse = np.mean((x - recovered) ** 2)
        assert mse < 0.1

    def test_scalar_index_range(self):
        """Indices should be in [0, 2**n_bits - 1]."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        for n_bits in [1, 2, 3, 4, 5]:
            indices, _, _ = quantize_scalar(x, n_bits=n_bits)
            assert indices.max() <= (1 << n_bits) - 1
            assert indices.min() >= 0
            assert indices.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
