"""Bridge CrowQuant compression into sqlite-vec databases."""
import sqlite3
import struct
import json
import numpy as np
from pathlib import Path
from .core import quantize, dequantize, serialize_block, deserialize_block, CrowQuantBlock


class SqliteVecBridge:
    """Compress and search vectors in sqlite-vec databases.

    Works with the sqlite-vec format used by crowclaw-memory, Claude Code
    memory, and Orion's memory engine. Embeddings are stored as float32
    blobs in a virtual table.
    """

    def __init__(self, db_path, table="chunks", vec_table="chunks_vec",
                 embedding_col="embedding", dim=768):
        """Connect to an existing sqlite-vec database.

        Args:
            db_path: Path to SQLite database.
            table: Main chunks table name.
            vec_table: Virtual vec0 table name.
            embedding_col: Column containing float32 embedding blobs.
            dim: Embedding dimensionality.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"database not found: {db_path}")
        self.table = table
        self.vec_table = vec_table
        self.embedding_col = embedding_col
        self.dim = dim
        self.conn = sqlite3.connect(str(self.db_path))

    def analyze(self, sample_size=1000):
        """Sample embeddings and analyze for compression profile.

        Returns:
            Dict with stats: count, dim, mean_norm, norm_std, size_bytes,
            recommended_bits, estimated_ratio.
        """
        cursor = self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table}"
        )
        total_count = cursor.fetchone()[0]

        cursor = self.conn.execute(
            f"SELECT {self.embedding_col} FROM {self.table} "
            f"WHERE {self.embedding_col} IS NOT NULL LIMIT ?",
            (sample_size,)
        )
        rows = cursor.fetchall()

        if not rows:
            return {"count": 0, "error": "no embeddings found"}

        norms = []
        actual_dim = None
        for (blob,) in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if actual_dim is None:
                actual_dim = len(vec)
            norms.append(np.linalg.norm(vec))

        norms = np.array(norms)
        mean_norm = float(norms.mean())
        norm_std = float(norms.std())
        size_bytes = total_count * actual_dim * 4  # float32

        # recommend bits based on norm variance
        if norm_std / mean_norm < 0.05:
            recommended_bits = 3  # very uniform -- aggressive ok
        elif norm_std / mean_norm < 0.15:
            recommended_bits = 4  # moderate variance
        else:
            recommended_bits = 5  # high variance -- be conservative

        return {
            "count": total_count,
            "sampled": len(rows),
            "dim": actual_dim,
            "mean_norm": mean_norm,
            "norm_std": norm_std,
            "norm_cv": float(norm_std / mean_norm) if mean_norm > 0 else 0,
            "original_size_bytes": size_bytes,
            "recommended_bits": recommended_bits,
            "estimated_ratio": 32.0 / recommended_bits,
        }

    def compress_database(self, n_bits=4, batch_size=100):
        """Compress all embeddings in the database.

        Creates a new table `{table}_cq` with compressed blobs alongside
        the original row IDs. Does NOT modify the original data.

        Args:
            n_bits: Bits per dimension.
            batch_size: Process this many rows at a time.

        Returns:
            Dict with compression stats.
        """
        cq_table = f"{self.table}_cq"

        self.conn.execute(f"DROP TABLE IF EXISTS {cq_table}")
        self.conn.execute(
            f"CREATE TABLE {cq_table} ("
            f"  rowid INTEGER PRIMARY KEY,"
            f"  compressed BLOB NOT NULL"
            f")"
        )

        cursor = self.conn.execute(
            f"SELECT rowid, {self.embedding_col} FROM {self.table} "
            f"WHERE {self.embedding_col} IS NOT NULL"
        )

        total = 0
        original_bytes = 0
        compressed_bytes = 0
        batch = []

        for row_id, blob in cursor:
            vec = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
            block = quantize(vec, n_bits=n_bits)
            serialized = serialize_block(block)

            batch.append((row_id, serialized))
            original_bytes += len(blob)
            compressed_bytes += len(serialized)
            total += 1

            if len(batch) >= batch_size:
                self.conn.executemany(
                    f"INSERT INTO {cq_table} (rowid, compressed) VALUES (?, ?)",
                    batch
                )
                batch = []

        if batch:
            self.conn.executemany(
                f"INSERT INTO {cq_table} (rowid, compressed) VALUES (?, ?)",
                batch
            )

        self.conn.commit()

        return {
            "vectors_compressed": total,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else 0,
            "table": cq_table,
        }

    def compressed_search(self, query_embedding, k=5):
        """Search using compressed representations.

        Decompresses each vector on the fly and computes cosine similarity.
        For small databases this is fast enough; for large ones, use an
        approximate index on top.

        Args:
            query_embedding: Query vector (numpy array).
            k: Number of results.

        Returns:
            List of (rowid, score) tuples.
        """
        cq_table = f"{self.table}_cq"
        query = np.asarray(query_embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        cursor = self.conn.execute(
            f"SELECT rowid, compressed FROM {cq_table}"
        )

        scores = []
        for row_id, blob in cursor:
            block = deserialize_block(blob)
            vec = dequantize(block)
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            scores.append((row_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def get_stats(self):
        """Compare original vs compressed storage."""
        cq_table = f"{self.table}_cq"

        try:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {cq_table}")
            cq_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            return {"error": f"compressed table '{cq_table}' not found. Run compress_database() first."}

        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {self.table}")
        orig_count = cursor.fetchone()[0]

        # estimate sizes
        cursor = self.conn.execute(
            f"SELECT SUM(LENGTH({self.embedding_col})) FROM {self.table} "
            f"WHERE {self.embedding_col} IS NOT NULL"
        )
        orig_bytes = cursor.fetchone()[0] or 0

        cursor = self.conn.execute(
            f"SELECT SUM(LENGTH(compressed)) FROM {cq_table}"
        )
        cq_bytes = cursor.fetchone()[0] or 0

        return {
            "original_count": orig_count,
            "compressed_count": cq_count,
            "original_bytes": orig_bytes,
            "compressed_bytes": cq_bytes,
            "ratio": orig_bytes / cq_bytes if cq_bytes > 0 else 0,
            "savings_pct": (1 - cq_bytes / orig_bytes) * 100 if orig_bytes > 0 else 0,
        }

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
