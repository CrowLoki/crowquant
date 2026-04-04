"""Bridge CrowQuant compression into Honcho memory databases."""
import sqlite3
import numpy as np
from pathlib import Path
from .core import quantize, dequantize, serialize_block, deserialize_block


class HonchoBridge:
    """Compress vectors in Hermes/Honcho state databases.

    Honcho is the memory server used by Hermes. Its state.db may contain
    embedding vectors for session memory. This bridge analyzes and
    compresses those vectors if present.
    """

    def __init__(self, state_db_path):
        """Connect to Hermes state.db.

        Args:
            state_db_path: Path to the Honcho state database.
        """
        self.db_path = Path(state_db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"state database not found: {state_db_path}")
        self.conn = sqlite3.connect(str(self.db_path))

    def _find_vector_columns(self):
        """Discover tables and columns that contain vector data.

        Returns:
            List of (table_name, column_name, sample_dim) tuples.
        """
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]

        vector_cols = []
        for table in tables:
            cursor = self.conn.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col_info in columns:
                col_name = col_info[1]
                col_type = (col_info[2] or "").upper()
                if col_type in ("BLOB", ""):
                    # check if it looks like a float32 vector
                    try:
                        cursor2 = self.conn.execute(
                            f"SELECT {col_name} FROM {table} "
                            f"WHERE {col_name} IS NOT NULL LIMIT 1"
                        )
                        row = cursor2.fetchone()
                        if row and row[0]:
                            blob = row[0]
                            if isinstance(blob, bytes) and len(blob) >= 8 and len(blob) % 4 == 0:
                                vec = np.frombuffer(blob, dtype=np.float32)
                                # heuristic: valid embedding has reasonable values
                                if len(vec) >= 64 and np.isfinite(vec).all():
                                    vector_cols.append((table, col_name, len(vec)))
                    except Exception:
                        continue

        return vector_cols

    def analyze(self, sample_size=100):
        """Analyze Honcho state for compressible vectors.

        Returns:
            Dict with analysis report including found vector columns
            and compression opportunities.
        """
        vector_cols = self._find_vector_columns()

        if not vector_cols:
            return {
                "found_vectors": False,
                "message": "no vector columns detected in state database",
                "tables_scanned": len(
                    self.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ),
            }

        results = []
        for table, col, dim in vector_cols:
            cursor = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL"
            )
            count = cursor.fetchone()[0]

            cursor = self.conn.execute(
                f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT ?",
                (sample_size,)
            )
            norms = []
            for (blob,) in cursor.fetchall():
                vec = np.frombuffer(blob, dtype=np.float32)
                norms.append(float(np.linalg.norm(vec)))

            norms = np.array(norms)
            results.append({
                "table": table,
                "column": col,
                "dim": dim,
                "count": count,
                "mean_norm": float(norms.mean()) if len(norms) > 0 else 0,
                "original_bytes": count * dim * 4,
                "estimated_4bit_bytes": count * dim // 2,
            })

        total_original = sum(r["original_bytes"] for r in results)
        total_compressed = sum(r["estimated_4bit_bytes"] for r in results)

        return {
            "found_vectors": True,
            "vector_columns": results,
            "total_original_bytes": total_original,
            "total_estimated_compressed": total_compressed,
            "estimated_ratio": total_original / total_compressed if total_compressed > 0 else 0,
        }

    def compress_sessions(self, n_bits=4):
        """Compress session vectors if present.

        Finds all vector columns and creates compressed copies in
        `{table}_cq` tables.

        Args:
            n_bits: Bits per dimension.

        Returns:
            Dict with per-table compression results.
        """
        vector_cols = self._find_vector_columns()
        if not vector_cols:
            return {"error": "no vector columns found to compress"}

        results = []
        for table, col, dim in vector_cols:
            cq_table = f"{table}_cq"
            self.conn.execute(f"DROP TABLE IF EXISTS {cq_table}")
            self.conn.execute(
                f"CREATE TABLE {cq_table} ("
                f"  rowid INTEGER PRIMARY KEY,"
                f"  compressed BLOB NOT NULL"
                f")"
            )

            cursor = self.conn.execute(
                f"SELECT rowid, {col} FROM {table} WHERE {col} IS NOT NULL"
            )

            total = 0
            orig_bytes = 0
            comp_bytes = 0
            batch = []

            for row_id, blob in cursor:
                vec = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
                block = quantize(vec, n_bits=n_bits)
                serialized = serialize_block(block)
                batch.append((row_id, serialized))
                orig_bytes += len(blob)
                comp_bytes += len(serialized)
                total += 1

                if len(batch) >= 100:
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

            results.append({
                "table": table,
                "column": col,
                "vectors_compressed": total,
                "original_bytes": orig_bytes,
                "compressed_bytes": comp_bytes,
                "ratio": orig_bytes / comp_bytes if comp_bytes > 0 else 0,
            })

        return {"compressed_tables": results}

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
