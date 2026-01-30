"""
Functional interface for Densatio pooling operations.

This module exposes all stateless pooling functions (1D, 2D, 3D).
These functions operate directly on tensors and are used internally by the
nn.Module layers, but can also be used standalone for functional API usage.
"""

# Import all functions from the 1D module
from .pooling_functions_1d import *

# Import all functions from the 2D module
from .pooling_functions_2d import *

# Import all functions from the 3D module
from .pooling_functions_3d import *
