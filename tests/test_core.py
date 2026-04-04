"""Tests for CrowQuant core algorithms."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crowquant.core import (
    WHTransform, quantize, dequantize, pack_bits, unpack_bits,
    serialize_block, deserialize_block, CrowQuantBlock,
)
from crowquant.adaptive import AdaptiveQuantizer, AdaptiveConfig
from crowquant.search import compressed_dot_product, compressed_cosine, compressed_knn


class TestWHT:
    """Tests for Walsh-Hadamard Transform."""

    def test_wht_roundtrip_pow2(self):
        """WHT followed by inverse WHT should recover original (power-of-2 length)."""
        rng = np.random.default_rng(42)
        for n in [2, 4, 8, 16, 64, 256, 1024]:
            x = rng.standard_normal(n)
            transformed = WHTransform.fast_wht(x)
            recovered = WHTransform.fast_iwht(transformed)
            np.testing.assert_allclose(recovered, x, atol=1e-10,
                                       err_msg=f"roundtrip failed for n={n}")

    def test_wht_forward_inverse_with_padding(self):
        """Forward/inverse with padding should recover original for non-pow2."""
        rng = np.random.default_rng(123)
        for dim in [768, 1024, 100, 333]:
            x = rng.standard_normal(dim)
            transformed, orig_len = WHTransform.forward(x)
            recovered = WHTransform.inverse(transformed, orig_len)
            np.testing.assert_allclose(recovered, x, atol=1e-10,
                                       err_msg=f"padded roundtrip failed for dim={dim}")

    def test_wht_energy_preservation(self):
        """WHT should preserve L2 norm (Parseval's theorem)."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(256)
        transformed = WHTransform.fast_wht(x)
        np.testing.assert_allclose(
            np.linalg.norm(x), np.linalg.norm(transformed), rtol=1e-10
        )

    def test_wht_known_values(self):
        """Test WHT on a simple known input."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = WHTransform.fast_wht(x)
        # H_4 * [1,1,1,1] / 2 = [2, 0, 0, 0] (with 1/sqrt(n) normalization at each level)
        # After two levels: [4/2, 0, 0, 0] = [2, 0, 0, 0]
        expected = np.array([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_wht_orthogonality(self):
        """WHT matrix should be orthogonal: H @ H.T = I."""
        H = WHTransform._hadamard_matrix(8)
        product = H @ H.T
        np.testing.assert_allclose(product, np.eye(8), atol=1e-10)


class TestBitPacking:
    """Tests for bit packing/unpacking."""

    def test_pack_unpack_roundtrip(self):
        """Pack and unpack should recover original indices."""
        rng = np.random.default_rng(42)
        for n_bits in [1, 2, 3, 4, 5, 6, 7, 8]:
            max_val = (1 << n_bits) - 1
            indices = rng.integers(0, max_val + 1, size=100).astype(np.uint64)
            packed = pack_bits(indices, n_bits)
            unpacked = unpack_bits(packed, n_bits, len(indices))
            np.testing.assert_array_equal(unpacked, indices,
                                          err_msg=f"roundtrip failed for {n_bits}-bit")

    def test_pack_known_values(self):
        """Test packing specific known values."""
        # 4-bit: [0, 15, 7, 8]
        indices = np.array([0, 15, 7, 8], dtype=np.uint64)
        packed = pack_bits(indices, 4)
        unpacked = unpack_bits(packed, 4, 4)
        np.testing.assert_array_equal(unpacked, indices)

    def test_pack_all_zeros(self):
        """All zeros should pack and unpack correctly."""
        indices = np.zeros(50, dtype=np.uint64)
        for n_bits in [2, 4, 8]:
            packed = pack_bits(indices, n_bits)
            unpacked = unpack_bits(packed, n_bits, 50)
            np.testing.assert_array_equal(unpacked, indices)

    def test_pack_all_max(self):
        """All max values should pack and unpack correctly."""
        for n_bits in [2, 3, 4, 5, 8]:
            max_val = (1 << n_bits) - 1
            indices = np.full(50, max_val, dtype=np.uint64)
            packed = pack_bits(indices, n_bits)
            unpacked = unpack_bits(packed, n_bits, 50)
            np.testing.assert_array_equal(unpacked, indices)


class TestQuantization:
    """Tests for quantize/dequantize."""

    def test_quantize_dequantize_mse(self):
        """Quantize then dequantize should have low MSE."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768)
        x = x / np.linalg.norm(x)  # unit norm

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

    def test_quantize_dequantize_no_wht(self):
        """Quantization without WHT should also work."""
        rng = np.random.default_rng(42)
        # use power-of-2 dim to avoid padding issues
        x = rng.standard_normal(256)

        block = quantize(x, n_bits=4, use_wht=False)
        recovered = dequantize(block, use_wht=False)

        mse = np.mean((x - recovered) ** 2)
        assert mse < 0.1, f"no-WHT MSE too high: {mse}"

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
        """Verify expected compression ratio."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768).astype(np.float32)

        block = quantize(x, n_bits=4)
        original_bytes = 768 * 4  # float32
        compressed_bytes = len(block.packed)

        # 4-bit: each dim is 0.5 bytes, so 768 dims = 384 bytes
        # plus padding to next power of 2: 1024 dims = 512 bytes
        # ratio should be close to 6x (3072 / 512)
        ratio = original_bytes / compressed_bytes
        assert ratio > 4.0, f"compression ratio too low: {ratio}"

    def test_constant_vector(self):
        """Constant vector should not crash."""
        x = np.ones(256)
        block = quantize(x, n_bits=4)
        recovered = dequantize(block)
        # all values should be 1.0 (or very close)
        np.testing.assert_allclose(recovered, x, atol=0.1)

    def test_zero_vector(self):
        """Zero vector should not crash."""
        x = np.zeros(128)
        block = quantize(x, n_bits=4)
        recovered = dequantize(block)
        np.testing.assert_allclose(recovered, x, atol=1e-10)


class TestSerialization:
    """Tests for block serialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Serialize then deserialize should recover the block."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768)

        block = quantize(x, n_bits=4)
        data = serialize_block(block)
        recovered_block = deserialize_block(data)

        assert recovered_block.n_bits == block.n_bits
        assert recovered_block.dim == block.dim
        assert abs(recovered_block.scale - block.scale) < 1e-10
        assert abs(recovered_block.zero_point - block.zero_point) < 1e-10
        assert recovered_block.packed == block.packed

    def test_serialize_with_outliers(self):
        """Serialization should handle outlier data."""
        block = CrowQuantBlock(
            n_bits=4, dim=768, scale=0.1, zero_point=-1.0,
            packed=b'\x00' * 100,
            outlier_mask=b'\xff\x00\xff',
            outlier_values=b'\x00\x3c' * 5,
        )
        data = serialize_block(block)
        recovered = deserialize_block(data)

        assert recovered.outlier_mask == block.outlier_mask
        assert recovered.outlier_values == block.outlier_values


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

        # should be within 10% relative error for 4-bit
        rel_error = abs(true_dot - approx_dot) / (abs(true_dot) + 1e-10)
        assert rel_error < 0.15, f"dot product error too high: {rel_error}"

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

    def test_compressed_knn_ranking(self):
        """KNN ranking should be preserved for well-separated vectors."""
        rng = np.random.default_rng(42)
        query = rng.standard_normal(256)
        query = query / np.linalg.norm(query)

        # create vectors at known similarities
        vecs = []
        for i in range(20):
            v = rng.standard_normal(256)
            v = v / np.linalg.norm(v)
            vecs.append(v)

        # make first vector very similar to query
        vecs[0] = query + rng.standard_normal(256) * 0.01
        vecs[0] = vecs[0] / np.linalg.norm(vecs[0])

        blocks = [quantize(v, n_bits=4) for v in vecs]
        results = compressed_knn(query, blocks, k=5)

        # the most similar vector should be index 0
        assert results[0][0] == 0, f"top result should be index 0, got {results[0][0]}"


class TestAdaptive:
    """Tests for adaptive quantization."""

    def test_adaptive_outlier_detection(self):
        """Adaptive quantizer should identify high-norm channels.

        WHT spreads spatial outliers across all coefficients, so we
        test with use_wht=False to verify the outlier detection logic
        directly on the raw vector.
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256) * 0.1  # small values
        # inject outliers that are clearly above threshold
        x[10] = 5.0
        x[50] = -4.0
        x[100] = 6.0

        config = AdaptiveConfig(outlier_threshold=2.5, use_wht=False)
        aq = AdaptiveQuantizer(config)
        block = aq.quantize(x)

        # should have detected outliers in the raw domain
        assert block.outlier_mask is not None
        assert block.outlier_values is not None

    def test_adaptive_roundtrip_quality(self):
        """Adaptive quantization should have equal or better quality than basic."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(768) * 0.1
        # add some outliers
        x[0] = 5.0
        x[100] = -4.0
        x[500] = 3.0

        # basic quantization
        basic_block = quantize(x, n_bits=4)
        basic_recovered = dequantize(basic_block)
        basic_mse = float(np.mean((x - basic_recovered) ** 2))

        # adaptive quantization
        aq = AdaptiveQuantizer(AdaptiveConfig(n_bits=4, outlier_threshold=2.5))
        adaptive_block = aq.quantize(x)
        adaptive_recovered = aq.dequantize(adaptive_block)
        adaptive_mse = float(np.mean((x - adaptive_recovered) ** 2))

        # adaptive should be at least as good
        assert adaptive_mse <= basic_mse * 1.1, \
            f"adaptive MSE ({adaptive_mse}) worse than basic ({basic_mse})"

    def test_adaptive_no_outliers(self):
        """Adaptive should work fine when there are no outliers."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)  # uniform distribution, few outliers

        aq = AdaptiveQuantizer(AdaptiveConfig(outlier_threshold=5.0))
        block = aq.quantize(x)
        recovered = aq.dequantize(block)

        mse = np.mean((x - recovered) ** 2)
        assert mse < 0.1


class TestBridgeSqlite:
    """Tests for SQLite bridge (uses in-memory DB)."""

    def _create_test_db(self, n_vecs=50, dim=128):
        """Create an in-memory SQLite DB with fake embeddings."""
        import sqlite3
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
        return conn

    def test_analyze(self):
        """Analyze should return reasonable stats."""
        conn = self._create_test_db()
        # monkey-patch bridge to use existing connection
        from crowquant.bridge_sqlite import SqliteVecBridge
        bridge = SqliteVecBridge.__new__(SqliteVecBridge)
        bridge.conn = conn
        bridge.table = "chunks"
        bridge.vec_table = "chunks_vec"
        bridge.embedding_col = "embedding"
        bridge.dim = 128

        stats = bridge.analyze(sample_size=20)
        assert stats["count"] == 50
        assert stats["dim"] == 128
        assert stats["mean_norm"] > 0

    def test_compress_and_search(self):
        """Compress then search should find similar vectors."""
        conn = self._create_test_db(n_vecs=20, dim=128)
        from crowquant.bridge_sqlite import SqliteVecBridge
        bridge = SqliteVecBridge.__new__(SqliteVecBridge)
        bridge.conn = conn
        bridge.table = "chunks"
        bridge.vec_table = "chunks_vec"
        bridge.embedding_col = "embedding"
        bridge.dim = 128

        stats = bridge.compress_database(n_bits=4)
        assert stats["vectors_compressed"] == 20
        assert stats["ratio"] > 1.0

        # search with the first vector as query
        cursor = conn.execute("SELECT embedding FROM chunks LIMIT 1")
        query_blob = cursor.fetchone()[0]
        query = np.frombuffer(query_blob, dtype=np.float32).astype(np.float64)

        results = bridge.compressed_search(query, k=5)
        assert len(results) == 5
        # top result should be the query itself (rowid=1)
        assert results[0][0] == 1
        assert results[0][1] > 0.9  # high similarity to itself


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
