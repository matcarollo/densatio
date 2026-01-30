from typing import Optional, Union, Tuple, Dict, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import pooling_functions_1d

__all__ = ["CustomPooling1d"]


class CustomPooling1d(nn.Module):
    """
    Custom 1D Pooling Layer for PyTorch.

    This layer provides a flexible interface for applying custom pooling operations defined
    in the functional module, supporting learnable parameters and various padding strategies.
    """

    def __init__(
            self,
            pool_size: Optional[int] = None,
            stride: Optional[int] = None,
            padding: Union[str, int, Tuple[int, int]] = 'valid',
            pooling_method: Union[str, Callable] = "max_pooling_1d",
            pooling_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Args:
            pool_size (int, optional): Size of the pooling window. Defaults to 2.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (str, int, or tuple): Padding strategy.
                - 'valid': No padding.
                - 'same': Padding to maintain size (if stride=1).
                - int: Symmetric padding.
                - tuple: Explicit padding (left, right).
            pooling_method (str or callable): Name of function in `pooling_functions_1d` or a callable.
                Defaults to 'max_pooling_1d'.
            pooling_params (dict, optional): Config for learnable parameters
                (keys: 'shape', 'init_value', 'requires_grad').
        """
        super(CustomPooling1d, self).__init__()

        self.pool_size = pool_size if pool_size is not None else 2
        self.stride = stride if stride is not None else 1

        if isinstance(pooling_method, str):
            if not hasattr(pooling_functions_1d, pooling_method):
                available = [name for name in dir(pooling_functions_1d) if not name.startswith('_')]
                raise ValueError(
                    f"Unknown pooling method: '{pooling_method}'. "
                    f"Available functions: {available}"
                )
            self.pooling_fn = getattr(pooling_functions_1d, pooling_method)
            self.pooling_method_name = pooling_method
        elif callable(pooling_method):
            self.pooling_fn = pooling_method
            self.pooling_method_name = pooling_method.__name__
        else:
            raise TypeError("pooling_method must be a string or a callable.")

        self.pooling_params = nn.ParameterDict()
        self.pooling_constants = {}

        if pooling_params is not None:
            for param_name, param_config in pooling_params.items():
                self._register_pooling_param(param_name, param_config)

        if isinstance(padding, str):
            if padding not in ['same', 'valid']:
                raise ValueError("If padding is a string, it must be 'same' or 'valid'.")
            self.padding_type = padding
            self.explicit_padding = None
        elif isinstance(padding, int):
            self.padding_type = 'custom'
            self.explicit_padding = (padding, padding)
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            self.padding_type = 'custom'
            self.explicit_padding = tuple(padding)
        else:
            raise ValueError("Padding must be 'same', 'valid', an int, or a 2-tuple.")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the pooling layer.

        Args:
            tensor (torch.Tensor): Input tensor of shape (Batch, Channels, Length).

        Returns:
            torch.Tensor: Pooled output tensor of shape (Batch, Channels, OutputLength).
        """
        if self.padding_type == 'same':
            padding = self._compute_same_padding(tensor.shape[-1], self.pool_size, self.stride)
        elif self.padding_type == 'valid':
            padding = (0, 0)
        else:
            padding = self.explicit_padding

        patches = self._extract_1d_patches(
            tensor,
            window_size=self.pool_size,
            stride=self.stride,
            padding=padding,
        )

        result = self._apply_pooling_method(patches)

        output_shape = self._get_output_shape_after_pooling(
            tensor.shape, self.pool_size, self.stride, padding
        )

        return result.view(*output_shape)

    def _register_pooling_param(self, param_name: str, param_config: Dict[str, Any]) -> None:
        """
        Registers a learnable parameter for the pooling operation.

        Args:
            param_name (str): The name of the parameter.
            param_config (dict): Configuration dictionary. Must contain:
                - 'shape': int or tuple defining the parameter shape.
                - 'init_value': float or torch.Tensor for initialization.
                - 'requires_grad' (optional): bool, defaults to True.

        Raises:
            TypeError: If param_config is not a dictionary.
            ValueError: If shape logic is inconsistent or types are invalid.
        """

        # 1. Strict type check: must be a dictionary
        if not isinstance(param_config, dict):
            raise TypeError(
                f"Param '{param_name}': Configuration must be a dictionary (e.g., {{'shape': (C, L), 'init_value': 0.5}}). "
                f"Got {type(param_config).__name__} instead."
            )

        shape_arg = param_config.get('shape')
        init_arg = param_config.get('init_value')
        requires_grad = param_config.get('requires_grad', True)

        if init_arg is None:
            init_arg = 0.5

        if shape_arg is not None:
            if isinstance(shape_arg, int):
                # (1, L): Shared across channels
                final_shape = (1, shape_arg)
            elif isinstance(shape_arg, (tuple, list)):
                # (C, L): Per-channel
                final_shape = tuple(shape_arg)
            else:
                raise ValueError(f"Param '{param_name}': 'shape' must be int or tuple.")

        elif isinstance(init_arg, torch.Tensor):
            final_shape = init_arg.shape
        else:
            raise ValueError(f"Param '{param_name}': You must provide 'shape' if 'init_value' is not a Tensor.")

        # --- Tensor Creation ---
        if isinstance(init_arg, torch.Tensor):
            if init_arg.shape != final_shape:
                raise ValueError(
                    f"Param '{param_name}': Init tensor shape {init_arg.shape} "
                    f"mismatches config shape {final_shape}."
                )
            data_tensor = init_arg.clone()

        elif isinstance(init_arg, (int, float)):
            data_tensor = torch.full(final_shape, fill_value=float(init_arg))

        else:
            raise TypeError(f"Param '{param_name}': 'init_value' type {type(init_arg)} not supported.")

        # --- Registration ---
        self.pooling_params[param_name] = nn.Parameter(
            data_tensor,
            requires_grad=requires_grad
        )

    def _apply_pooling_method(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Applies the configured pooling function to extracted patches.
        """
        kwargs = {**self.pooling_params, **self.pooling_constants}
        return self.pooling_fn(patches, **kwargs).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _compute_same_padding(length: int, window_size: int, stride: int) -> Tuple[int, int]:
        """Computes symmetric padding to maintain input size (if stride=1)."""
        out_len = (length + stride - 1) // stride
        pad_needed = max((out_len - 1) * stride + window_size - length, 0)
        left = pad_needed // 2
        right = pad_needed - left
        return (left, right)

    @staticmethod
    def _extract_1d_patches(tensor: torch.Tensor, window_size: int, stride: int,
                            padding: Tuple[int, int]) -> torch.Tensor:
        """Extracts sliding windows from the 1D input tensor."""
        if padding and (padding[0] > 0 or padding[1] > 0):
            tensor = F.pad(tensor, (padding[0], padding[1]))
        patches = tensor.unfold(dimension=2, size=window_size, step=stride)
        return patches.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _get_output_shape_after_pooling(input_shape: Tuple[int, ...], kernel_size: int,
                                        stride: int, padding: Tuple[int, int]) -> Tuple[int, int, int]:
        """Calculates the expected output shape."""
        batch_size, channels, length = input_shape
        total_pad = sum(padding)
        output_length = (length + total_pad - kernel_size) // stride + 1
        return batch_size, channels, output_length
