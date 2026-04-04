"""CrowQuant -- Adaptive vector compression for the CrowClaw ecosystem."""
__version__ = "0.1.0"
from .core import quantize, dequantize, WHTransform
from .adaptive import AdaptiveQuantizer
from .search import compressed_dot_product, compressed_knn
from .profiles import get_profile, CrowStation, Universal
