"""
Densatio: A PyTorch Library for Customizable Pooling Operations.

This package provides highly configurable pooling layers for neural networks,
supporting 1D, 2D, and 3D operations with both built-in and user-defined
aggregation functions. It bridges the gap between standard pooling layers
and research-grade custom implementations.
"""

__version__ = "0.1.0"
__author__ = "Matteo Carollo"
__email__ = "mcarollo@unisa.it"

# 1. Import Main Layers (The primary API for most users)
# We import from the .layers subpackage where the nn.Modules reside.
from .layers.custom_pooling_1d import CustomPooling1d
from .layers.custom_pooling_2d import CustomPooling2d
from .layers.custom_pooling_3d import CustomPooling3d

# 2. Expose the functional API (For advanced usage / custom implementations)
# This allows users to access raw functions like F.max_pooling_1d directly.
from . import functional

# Public API definition
__all__ = [
    # Main Layer Classes
    "CustomPooling1d",
    "CustomPooling2d",
    "CustomPooling3d",

    # Functional Module
    "functional",

    # Metadata
    "__version__",
    "__author__",
]