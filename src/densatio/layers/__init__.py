"""
Densatio Layers Module.

This package contains the stateful nn.Module implementations of the custom pooling layers.
These classes wrap the functional operations and handle parameter management (if any)
and input/output shaping.
"""

from .custom_pooling_1d import CustomPooling1d
from .custom_pooling_2d import CustomPooling2d
from .custom_pooling_3d import CustomPooling3d

__all__ = [
    "CustomPooling1d",
    "CustomPooling2d",
    "CustomPooling3d",
]
