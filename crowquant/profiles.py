"""Compression profiles for different hardware targets."""
from dataclasses import dataclass


@dataclass
class CompressionProfile:
    """A named compression configuration."""
    name: str
    n_bits: int
    use_wht: bool
    adaptive_outliers: bool
    outlier_threshold: float
    description: str

    @property
    def theoretical_ratio(self) -> float:
        """Theoretical compression ratio vs float32."""
        # float32 = 32 bits per dim
        # compressed = n_bits per dim (ignoring small overhead)
        return 32.0 / self.n_bits


# CrowStation profile: aggressive compression for Crow's local hardware
# (40GB RAM, RTX 3050 Ti 4GB VRAM, 2.5TB NVMe)
CrowStation = CompressionProfile(
    name="CrowStation",
    n_bits=3,
    use_wht=True,
    adaptive_outliers=True,
    outlier_threshold=2.5,
    description=(
        "Aggressive 3-bit compression with WHT and outlier preservation. "
        "Optimized for Crow's local machine -- fits ~10x more vectors in "
        "the same memory footprint with <5% recall loss."
    ),
)

# Universal profile: safe default that works everywhere
Universal = CompressionProfile(
    name="Universal",
    n_bits=4,
    use_wht=True,
    adaptive_outliers=True,
    outlier_threshold=3.0,
    description=(
        "Balanced 4-bit compression with WHT. Works well on any hardware. "
        "8x compression with <2% recall loss on typical embeddings."
    ),
)

# Additional profiles
Aggressive = CompressionProfile(
    name="Aggressive",
    n_bits=2,
    use_wht=True,
    adaptive_outliers=True,
    outlier_threshold=2.0,
    description=(
        "Maximum 2-bit compression for extremely memory-constrained "
        "environments. 16x compression with ~10-15% recall loss."
    ),
)

HighFidelity = CompressionProfile(
    name="HighFidelity",
    n_bits=8,
    use_wht=True,
    adaptive_outliers=False,
    outlier_threshold=3.0,
    description=(
        "Conservative 8-bit compression. 4x compression with negligible "
        "recall loss. Good for critical vectors where accuracy matters."
    ),
)

_PROFILES = {
    "crowstation": CrowStation,
    "universal": Universal,
    "aggressive": Aggressive,
    "highfidelity": HighFidelity,
}


def get_profile(name: str) -> CompressionProfile:
    """Get a compression profile by name (case-insensitive)."""
    key = name.lower()
    if key not in _PROFILES:
        available = ", ".join(_PROFILES.keys())
        raise ValueError(f"unknown profile '{name}', available: {available}")
    return _PROFILES[key]


def list_profiles() -> list[CompressionProfile]:
    """Return all available profiles."""
    return list(_PROFILES.values())
