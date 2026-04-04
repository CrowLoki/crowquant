"""CrowQuant CLI -- compress, analyze, benchmark vectors."""
import argparse
import sys
import time
import numpy as np
from pathlib import Path


def cmd_compress(args):
    """Compress vectors in a database."""
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"error: {db_path} not found")
        sys.exit(1)

    if args.type == "sqlite":
        from .bridge_sqlite import SqliteVecBridge
        bridge = SqliteVecBridge(db_path)
        print(f"compressing {db_path} (sqlite, {args.bits}-bit)...")
        stats = bridge.compress_database(n_bits=args.bits)
        bridge.close()
    elif args.type == "lance":
        from .bridge_lance import LanceBridge
        bridge = LanceBridge(db_path)
        tables = bridge.list_tables()
        if not tables:
            print("error: no tables found in LanceDB")
            sys.exit(1)
        print(f"compressing {db_path} (lance, {args.bits}-bit)...")
        stats = {}
        for t in tables:
            if t.endswith("_cq"):
                continue
            try:
                s = bridge.compress_table(t, n_bits=args.bits)
                stats[t] = s
                print(f"  {t}: {s['vectors_compressed']} vectors, {s['ratio']:.1f}x ratio")
            except Exception as e:
                print(f"  {t}: skipped ({e})")
    elif args.type == "honcho":
        from .bridge_honcho import HonchoBridge
        bridge = HonchoBridge(db_path)
        print(f"compressing {db_path} (honcho, {args.bits}-bit)...")
        stats = bridge.compress_sessions(n_bits=args.bits)
        bridge.close()
    else:
        print(f"error: unknown type '{args.type}'")
        sys.exit(1)

    print(f"\ndone. stats: {_format_stats(stats)}")


def cmd_analyze(args):
    """Analyze a database for compression potential."""
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"error: {db_path} not found")
        sys.exit(1)

    # auto-detect type
    if db_path.suffix == ".db" or db_path.suffix == ".sqlite":
        # try honcho first (it auto-detects vector columns)
        from .bridge_honcho import HonchoBridge
        try:
            bridge = HonchoBridge(db_path)
            analysis = bridge.analyze()
            bridge.close()
            if analysis.get("found_vectors"):
                _print_analysis("honcho/sqlite", analysis)
                return
        except Exception:
            pass

        # fall back to sqlite-vec
        from .bridge_sqlite import SqliteVecBridge
        try:
            bridge = SqliteVecBridge(db_path)
            analysis = bridge.analyze()
            bridge.close()
            _print_analysis("sqlite-vec", analysis)
            return
        except Exception as e:
            print(f"error analyzing: {e}")
            sys.exit(1)
    elif db_path.is_dir():
        from .bridge_lance import LanceBridge
        try:
            bridge = LanceBridge(db_path)
            tables = bridge.list_tables()
            for t in tables:
                if t.endswith("_cq"):
                    continue
                try:
                    analysis = bridge.analyze(t)
                    _print_analysis(f"lance/{t}", analysis)
                except Exception as e:
                    print(f"  {t}: skipped ({e})")
            return
        except Exception as e:
            print(f"error analyzing: {e}")
            sys.exit(1)

    print(f"error: could not detect database type for {db_path}")
    sys.exit(1)


def cmd_benchmark(args):
    """Run compression benchmarks."""
    from .core import quantize, dequantize
    from .adaptive import AdaptiveQuantizer, AdaptiveConfig
    from .search import compressed_dot_product

    dim = args.dim
    count = args.count
    n_bits = args.bits

    print(f"CrowQuant benchmark")
    print(f"  dim={dim}, count={count}, bits={n_bits}")
    print()

    # generate random embeddings (simulating normalized sentence embeddings)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((count, dim)).astype(np.float64)
    # normalize to unit length (typical for sentence embeddings)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    # basic quantize/dequantize
    print("--- basic quantization ---")
    t0 = time.perf_counter()
    blocks = [quantize(v, n_bits=n_bits) for v in vecs]
    t_quant = time.perf_counter() - t0
    print(f"  quantize: {t_quant:.3f}s ({count / t_quant:.0f} vec/s)")

    t0 = time.perf_counter()
    recovered = np.stack([dequantize(b) for b in blocks])
    t_dequant = time.perf_counter() - t0
    print(f"  dequantize: {t_dequant:.3f}s ({count / t_dequant:.0f} vec/s)")

    # MSE
    mse = float(np.mean((vecs - recovered) ** 2))
    print(f"  MSE: {mse:.6f}")

    # cosine similarity preservation
    n_pairs = min(1000, count)
    idx_a = rng.integers(0, count, n_pairs)
    idx_b = rng.integers(0, count, n_pairs)
    true_cos = np.array([
        np.dot(vecs[a], vecs[b]) for a, b in zip(idx_a, idx_b)
    ])
    approx_cos = np.array([
        np.dot(recovered[a], recovered[b]) /
        (np.linalg.norm(recovered[a]) * np.linalg.norm(recovered[b]) + 1e-10)
        for a, b in zip(idx_a, idx_b)
    ])
    cos_error = float(np.mean(np.abs(true_cos - approx_cos)))
    print(f"  mean |cosine error|: {cos_error:.6f}")

    # compression ratio
    orig_bytes = count * dim * 4
    comp_bytes = sum(len(b.packed) for b in blocks)
    # add overhead estimate (header per block)
    overhead = count * 26  # ~26 bytes header per block
    total_comp = comp_bytes + overhead
    print(f"  compression: {orig_bytes:,} -> {total_comp:,} bytes ({orig_bytes / total_comp:.1f}x)")

    # adaptive quantization
    print()
    print("--- adaptive quantization ---")
    aq = AdaptiveQuantizer(AdaptiveConfig(n_bits=n_bits))
    t0 = time.perf_counter()
    adaptive_blocks = [aq.quantize(v) for v in vecs[:1000]]
    t_adaptive = time.perf_counter() - t0
    adaptive_recovered = np.stack([aq.dequantize(b) for b in adaptive_blocks])
    adaptive_mse = float(np.mean((vecs[:1000] - adaptive_recovered) ** 2))
    n_with_outliers = sum(1 for b in adaptive_blocks if b.outlier_mask is not None)
    print(f"  quantize (1000): {t_adaptive:.3f}s")
    print(f"  MSE: {adaptive_mse:.6f}")
    print(f"  blocks with outliers: {n_with_outliers}/1000")

    print()
    print("done.")


def cmd_info(args):
    """Show CrowQuant info and detected hardware."""
    from . import __version__
    from .profiles import list_profiles

    print(f"CrowQuant v{__version__}")
    print()

    # hardware
    print("hardware:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU: {gpu} ({vram:.1f} GB VRAM)")
        else:
            print("  GPU: none (CPU only)")
    except ImportError:
        print("  GPU: torch not installed (CPU only)")

    import os
    cpu_count = os.cpu_count() or 1
    print(f"  CPU cores: {cpu_count}")

    # numpy info
    print(f"  numpy: {np.__version__}")

    # profiles
    print()
    print("profiles:")
    for p in list_profiles():
        print(f"  {p.name} ({p.n_bits}-bit): {p.description}")

    # optional deps
    print()
    print("optional dependencies:")
    for pkg in ["torch", "lancedb", "sqlite_vec"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: not installed")


def _format_stats(stats):
    """Format stats dict for display."""
    if isinstance(stats, dict):
        parts = []
        for k, v in stats.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)
    return str(stats)


def _print_analysis(db_type, analysis):
    """Pretty-print analysis results."""
    print(f"\nanalysis ({db_type}):")
    for k, v in analysis.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    {item}")
        else:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="CrowQuant -- Adaptive Vector Compression"
    )
    subparsers = parser.add_subparsers()

    # crowquant compress <db_path> [--bits 3] [--type sqlite|lance|honcho]
    p_compress = subparsers.add_parser(
        "compress", help="compress vectors in a database"
    )
    p_compress.add_argument("db_path", help="path to database")
    p_compress.add_argument("--bits", type=int, default=4, help="bits per dim (default: 4)")
    p_compress.add_argument(
        "--type", choices=["sqlite", "lance", "honcho"], default="sqlite",
        help="database type (default: sqlite)"
    )
    p_compress.set_defaults(func=cmd_compress)

    # crowquant analyze <db_path>
    p_analyze = subparsers.add_parser(
        "analyze", help="analyze a database for compression potential"
    )
    p_analyze.add_argument("db_path", help="path to database")
    p_analyze.set_defaults(func=cmd_analyze)

    # crowquant benchmark [--dim 768] [--count 10000] [--bits 3]
    p_bench = subparsers.add_parser(
        "benchmark", help="run compression benchmarks"
    )
    p_bench.add_argument("--dim", type=int, default=768, help="vector dimension (default: 768)")
    p_bench.add_argument("--count", type=int, default=10000, help="number of vectors (default: 10000)")
    p_bench.add_argument("--bits", type=int, default=4, help="bits per dim (default: 4)")
    p_bench.set_defaults(func=cmd_benchmark)

    # crowquant info
    p_info = subparsers.add_parser(
        "info", help="show CrowQuant info and hardware"
    )
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
