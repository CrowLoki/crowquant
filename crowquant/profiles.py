"""Hardware profiles for CrowQuant.

Defines preset configurations optimised for different hardware targets.
Auto-detection selects the best profile based on available resources.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareProfile:
    """Configuration profile for a specific hardware target.

    Attributes
    ----------
    name : str
        Human-readable profile name.
    default_bits : int
        Default quantization bit-width.
    use_cuda : bool
        Whether to use CUDA-accelerated operations.
    batch_size : int
        Recommended batch size for bulk operations.
    wht_method : str
        WHT implementation to use ("numpy" or "cuda").
    description : str
        Human-readable description of the target hardware.

    Examples
    --------
    >>> p = HardwareProfile("Test", 4, False, 64, "numpy", "Test profile")
    >>> p.name
    'Test'
    """

    name: str
    default_bits: int
    use_cuda: bool
    batch_size: int
    wht_method: str
    description: str


CrowStation = HardwareProfile(
    name="CrowStation",
    default_bits=3,
    use_cuda=True,
    batch_size=512,
    wht_method="cuda",
    description="Crow's ASUS TUF Dash F15 (RTX 3050 Ti, 40GB RAM)",
)

Universal = HardwareProfile(
    name="Universal",
    default_bits=4,
    use_cuda=False,
    batch_size=128,
    wht_method="numpy",
    description="Any machine, pure Python + NumPy",
)


def get_profile(name: str | None = None) -> HardwareProfile:
    """Auto-detect or select a hardware profile.

    Parameters
    ----------
    name : str or None
        Profile name to select.  Recognised values: "crowstation",
        "universal".  If None, auto-detects based on CUDA availability.

    Returns
    -------
    HardwareProfile
        The selected or auto-detected profile.

    Raises
    ------
    ValueError
        If name is not a recognised profile.

    Examples
    --------
    >>> get_profile("universal").name
    'Universal'
    >>> get_profile("crowstation").use_cuda
    True
    >>> get_profile().name in ("CrowStation", "Universal")
    True
    """
    if name is not None:
        key = name.lower().strip()
        if key == "crowstation":
            return CrowStation
        if key == "universal":
            return Universal
        raise ValueError(
            f"Unknown profile '{name}'. Choose 'crowstation' or 'universal'."
        )

    # Auto-detect
    try:
        import torch

        if torch.cuda.is_available():
            return CrowStation
    except ImportError:
        pass

    return Universal
