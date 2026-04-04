# CrowQuant

Adaptive vector compression for AI memory systems. Shrinks embedding databases 6-10x while preserving search quality.

CrowQuant uses Walsh-Hadamard Transform (WHT) to decorrelate embedding dimensions before quantizing them to low bit-widths (2-8 bits). An adaptive outlier detector preserves the high-magnitude channels that matter most for similarity search.

## Quick Start

```bash
pip install -e .

# analyze a database
crowquant analyze path/to/memory.db

# compress it
crowquant compress path/to/memory.db --bits 4

# run benchmarks
crowquant benchmark --dim 768 --count 10000 --bits 4

# show info
crowquant info
```

## Compression Profiles

### CrowStation (3-bit)

Aggressive compression for local hardware with plenty of disk but limited VRAM. ~10x compression with <5% recall loss. Uses WHT + adaptive outlier preservation at a lower threshold.

### Universal (4-bit)

Safe default that works everywhere. ~8x compression with <2% recall loss on typical sentence embeddings. Good starting point for any deployment.

### Additional Profiles

- **Aggressive** (2-bit) -- maximum compression, 16x ratio, ~10-15% recall loss
- **HighFidelity** (8-bit) -- conservative, 4x ratio, negligible recall loss

## Algorithm

1. **Walsh-Hadamard Transform** -- decorrelates embedding dimensions using an orthogonal +1/-1 transform. O(n log n) via butterfly operations. Energy concentrates into fewer coefficients.

2. **Adaptive Outlier Detection** -- channels with magnitude > threshold * std get stored at float16. These carry disproportionate information for similarity ranking.

3. **Uniform Quantization** -- remaining channels are uniformly quantized to n-bit integers with a global scale and zero-point per vector.

4. **Bit Packing** -- quantized indices are packed into bytes at arbitrary bit-widths (not just powers of 2).

## Integration

### SQLite-vec (Claude Code, CrowClaw, Orion memory)

```python
from crowquant.bridge_sqlite import SqliteVecBridge

with SqliteVecBridge("~/.claude/memory/claude_memory.sqlite") as bridge:
    stats = bridge.analyze()
    print(stats)
    result = bridge.compress_database(n_bits=4)
    print(f"compressed {result['vectors_compressed']} vectors, {result['ratio']:.1f}x")
```

### LanceDB (Orion vectors)

```python
from crowquant.bridge_lance import LanceBridge

bridge = LanceBridge("path/to/lance_db")
bridge.compress_table("embeddings", n_bits=4)
```

### Honcho (Hermes memory)

```python
from crowquant.bridge_honcho import HonchoBridge

with HonchoBridge("path/to/state.db") as bridge:
    analysis = bridge.analyze()
    if analysis["found_vectors"]:
        bridge.compress_sessions(n_bits=4)
```

### Python API

```python
import numpy as np
from crowquant import quantize, dequantize, WHTransform
from crowquant.adaptive import AdaptiveQuantizer
from crowquant.search import compressed_cosine, compressed_knn
from crowquant.profiles import get_profile

# basic compression
vec = np.random.randn(768).astype(np.float32)
block = quantize(vec, n_bits=4)
recovered = dequantize(block)

# adaptive compression (outlier-aware)
aq = AdaptiveQuantizer()
block = aq.quantize(vec)
recovered = aq.dequantize(block)

# search
blocks = [quantize(v, n_bits=4) for v in vectors]
results = compressed_knn(query_vec, blocks, k=10)

# profiles
profile = get_profile("crowstation")
print(f"{profile.name}: {profile.n_bits}-bit, {profile.theoretical_ratio}x ratio")
```

## CLI Usage

```
crowquant compress <db_path> [--bits N] [--type sqlite|lance|honcho]
crowquant analyze <db_path>
crowquant benchmark [--dim 768] [--count 10000] [--bits 4]
crowquant info
```

## Benchmarks

Run `crowquant benchmark` to generate benchmarks for your hardware. Typical results on an i7-11370H:

| Bits | Dim  | Ratio | MSE       | Cosine Error | Quantize   | Dequantize |
|------|------|-------|-----------|--------------|------------|------------|
| 4    | 768  | ~6x   | ~0.001    | ~0.003       | ~5k vec/s  | ~8k vec/s  |
| 3    | 768  | ~8x   | ~0.003    | ~0.008       | ~5k vec/s  | ~8k vec/s  |
| 2    | 768  | ~12x  | ~0.01     | ~0.02        | ~5k vec/s  | ~8k vec/s  |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
