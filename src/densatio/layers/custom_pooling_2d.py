from typing import Optional, Union, Tuple, Dict, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import pooling_functions_2d

__all__ = ["CustomPooling2d"]


class CustomPooling2d(nn.Module):
    """
    Custom 2D Pooling Layer for PyTorch.

    This layer provides a flexible interface for applying custom pooling operations defined
    in the functional module, supporting learnable parameters and various padding strategies.
    """

    def __init__(
            self,
            pool_size: Optional[Union[int, Tuple[int, int]]] = None,
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Union[str, int, Tuple[int, int, int, int]] = 'valid',
            pooling_method: Union[str, Callable] = "max_pooling_2d",
            pooling_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Args:
            pool_size (int or tuple, optional): Size of the pooling window (H, W). Defaults to (2, 2).
            stride (int or tuple, optional): Stride of the pooling operation (H, W). Defaults to pool_size.
            padding (str, int, or tuple): Padding strategy.
                - 'valid': No padding.
                - 'same': Padding to maintain size (if stride=1).
                - int: Symmetric padding.
                - tuple: Explicit padding (left, right, top, bottom).
            pooling_method (str or callable): Name of function in `pooling_functions_2d` or a callable.
                Defaults to 'max_pooling_2d'.
            pooling_params (dict, optional): Config for learnable parameters
                (keys: 'shape', 'init_value', 'requires_grad').
        """
        super(CustomPooling2d, self).__init__()

        if pool_size is None:
            self.pool_size = (2, 2)
        elif isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
            self.pool_size = tuple(pool_size)
        else:
            raise ValueError("pool_size must be int or 2-tuple")

        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, (tuple, list)) and len(stride) == 2:
            self.stride = tuple(stride)
        else:
            raise ValueError("stride must be int or 2-tuple")

        if isinstance(pooling_method, str):
            if not hasattr(pooling_functions_2d, pooling_method):
                available = [name for name in dir(pooling_functions_2d) if not name.startswith('_')]
                raise ValueError(
                    f"Unknown pooling method: '{pooling_method}'. "
                    f"Available functions: {available}"
                )
            self.pooling_fn = getattr(pooling_functions_2d, pooling_method)
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
            self.explicit_padding = (padding, padding, padding, padding)
        elif isinstance(padding, (tuple, list)) and len(padding) == 4:
            self.padding_type = 'custom'
            self.explicit_padding = tuple(padding)
        else:
            raise ValueError("Padding must be 'same', 'valid', an int, or a 4-tuple (left, right, top, bottom).")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 2D pooling layer.

        Args:
            tensor (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Pooled output tensor of shape (Batch, Channels, Height_out, Width_out).
        """
        if self.padding_type == 'same':
            padding = self._compute_same_padding(tensor.shape[-2:], self.pool_size, self.stride)
        elif self.padding_type == 'valid':
            padding = (0, 0, 0, 0)
        else:
            padding = self.explicit_padding

        patches = self._extract_2d_patches(
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
        Registers a learnable parameter for 2D pooling.

        Args:
            param_name (str): The name of the parameter.
            param_config (dict): Configuration dictionary.
                Shape Logic for 2D:
                - int: (1, H, W) [Shared]
                - tuple(C, H, W) [Per-Channel]
        """

        # 1. Strict type check: must be a dictionary
        if not isinstance(param_config, dict):
            raise TypeError(
                f"Param '{param_name}': Configuration must be a dictionary (e.g., {{'shape': (C, H, W), 'init_value': 0.5}}). "
                f"Got {type(param_config).__name__} instead."
            )

        shape_arg = param_config.get('shape')
        init_arg = param_config.get('init_value')
        requires_grad = param_config.get('requires_grad', True)

        if init_arg is None:
            init_arg = 0.5

        # --- Final Shape Determination ---
        if shape_arg is not None:
            if isinstance(shape_arg, int):
                # (1, S, S) Shared Square
                final_shape = (1, shape_arg, shape_arg)
            elif isinstance(shape_arg, (tuple, list)):
                if len(shape_arg) == 2:
                    # (1, H, W) Shared Rect
                    final_shape = (1, shape_arg[0], shape_arg[1])
                elif len(shape_arg) == 3:
                    # (C, H, W) Per-Channel
                    final_shape = tuple(shape_arg)
                else:
                    raise ValueError(f"Param '{param_name}': tuple shape must be length 2 (H,W) or 3 (C,H,W).")
            else:
                raise ValueError(f"Param '{param_name}': shape must be int or tuple.")

        elif isinstance(init_arg, torch.Tensor):
            final_shape = init_arg.shape
        else:
            raise ValueError(f"Param '{param_name}': Provide 'shape' if 'init_value' is not Tensor.")

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
        Applies the configured pooling function to extracted 2D patches.
        """
        kwargs = {**self.pooling_params, **self.pooling_constants}
        # Maintains original permutation logic
        return self.pooling_fn(patches, **kwargs).permute(0, 3, 1, 2, 4, 5).contiguous()

    @staticmethod
    def _compute_same_padding(input_size: Tuple[int, int], kernel_size: Tuple[int, int],
                              stride: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Computes symmetric padding for height and width."""
        h, w = input_size
        k_h, k_w = kernel_size
        s_h, s_w = stride

        out_h = (h + s_h - 1) // s_h
        out_w = (w + s_w - 1) // s_w

        pad_h = max((out_h - 1) * s_h + k_h - h, 0)
        pad_w = max((out_w - 1) * s_w + k_w - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return (pad_left, pad_right, pad_top, pad_bottom)

    @staticmethod
    def _extract_2d_patches(tensor: torch.Tensor, window_size: Tuple[int, int],
                            stride: Tuple[int, int], padding: Tuple[int, int, int, int]) -> torch.Tensor:
        """Extracts sliding windows from the 2D input tensor."""
        if padding and any(p > 0 for p in padding):
            tensor = F.pad(tensor, padding)

        patches = tensor.unfold(2, window_size[0], stride[0])
        patches = patches.unfold(3, window_size[1], stride[1])

        # Maintains original permutation logic
        return patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    @staticmethod
    def _get_output_shape_after_pooling(input_shape: Tuple[int, ...], kernel_size: Tuple[int, int],
                                        stride: Tuple[int, int], padding: Tuple[int, int, int, int]) -> Tuple[
        int, int, int, int]:
        """Calculates the expected output shape."""
        batch_size, channels, height, width = input_shape
        pad_h = padding[2] + padding[3]
        pad_w = padding[0] + padding[1]

        out_height = (height + pad_h - kernel_size[0]) // stride[0] + 1
        out_width = (width + pad_w - kernel_size[1]) // stride[1] + 1

        return batch_size, channels, out_height, out_width
