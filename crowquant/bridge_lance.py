"""Bridge CrowQuant compression into LanceDB databases."""
import numpy as np
from pathlib import Path
from .core import quantize, dequantize, serialize_block, deserialize_block


class LanceBridge:
    """Compress and manage vectors in LanceDB tables.

    LanceDB stores vectors in Lance columnar format. This bridge adds
    compressed representations alongside the originals.
    """

    def __init__(self, db_path):
        """Connect to a LanceDB database.

        Args:
            db_path: Path to LanceDB directory.
        """
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb required for LanceBridge: pip install lancedb"
            )
        self.db_path = Path(db_path)
        self.db = lancedb.connect(str(self.db_path))

    def list_tables(self):
        """List all tables in the database."""
        return self.db.table_names()

    def analyze(self, table_name, embedding_col="embedding", sample_size=1000):
        """Analyze LanceDB vectors for compression profile.

        Args:
            table_name: Name of the LanceDB table.
            embedding_col: Column with vector data.
            sample_size: Number of vectors to sample.

        Returns:
            Dict with analysis stats.
        """
        table = self.db.open_table(table_name)
        df = table.to_pandas()

        total_count = len(df)
        if total_count == 0:
            return {"count": 0, "error": "empty table"}

        sample_df = df.head(sample_size)
        embeddings = np.stack(sample_df[embedding_col].values)
        dim = embeddings.shape[1]
        norms = np.linalg.norm(embeddings, axis=1)

        mean_norm = float(norms.mean())
        norm_std = float(norms.std())
        size_bytes = total_count * dim * 4

        if norm_std / mean_norm < 0.05:
            recommended_bits = 3
        elif norm_std / mean_norm < 0.15:
            recommended_bits = 4
        else:
            recommended_bits = 5

        return {
            "count": total_count,
            "sampled": len(sample_df),
            "dim": dim,
            "mean_norm": mean_norm,
            "norm_std": norm_std,
            "norm_cv": float(norm_std / mean_norm) if mean_norm > 0 else 0,
            "original_size_bytes": size_bytes,
            "recommended_bits": recommended_bits,
            "estimated_ratio": 32.0 / recommended_bits,
        }

    def compress_table(self, table_name, embedding_col="embedding", n_bits=4):
        """Add compressed column to LanceDB table.

        Creates a new table `{table_name}_cq` with the compressed embeddings
        stored as binary blobs alongside the original row indices.

        Args:
            table_name: Source table.
            embedding_col: Column with vectors.
            n_bits: Bits per dimension.

        Returns:
            Dict with compression stats.
        """
        import pyarrow as pa

        table = self.db.open_table(table_name)
        df = table.to_pandas()

        compressed_blobs = []
        original_bytes = 0
        compressed_bytes = 0

        for vec in df[embedding_col].values:
            vec = np.asarray(vec, dtype=np.float64)
            original_bytes += vec.size * 4  # float32 equivalent
            block = quantize(vec, n_bits=n_bits)
            blob = serialize_block(block)
            compressed_blobs.append(blob)
            compressed_bytes += len(blob)

        # store as a new table with id + compressed blob
        cq_table_name = f"{table_name}_cq"
        cq_data = pa.table({
            "id": pa.array(range(len(df))),
            "compressed": pa.array(compressed_blobs, type=pa.binary()),
        })

        # drop if exists and recreate
        try:
            self.db.drop_table(cq_table_name)
        except Exception:
            pass
        self.db.create_table(cq_table_name, cq_data)

        return {
            "vectors_compressed": len(df),
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else 0,
            "table": cq_table_name,
        }

    def compressed_search(self, table_name, query_embedding, k=5):
        """Search compressed table by cosine similarity.

        Args:
            table_name: Original table name (will look for {name}_cq).
            query_embedding: Query vector.
            k: Number of results.

        Returns:
            List of (id, score) tuples.
        """
        cq_table_name = f"{table_name}_cq"
        table = self.db.open_table(cq_table_name)
        df = table.to_pandas()

        query = np.asarray(query_embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        scores = []
        for _, row in df.iterrows():
            block = deserialize_block(row["compressed"])
            vec = dequantize(block)
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            scores.append((int(row["id"]), score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
