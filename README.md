# Densatio

**Densatio** is a PyTorch library designed to streamline the creation and management of **custom pooling layers** for **1D, 2D, and 3D data**.

It abstracts away the complexity of sliding windows (kernel, stride, padding, etc.) across multiple dimensions, allowing researchers to focus solely on the pooling logic.

> **Research Origin:** This library implements the generalized pooling functions presented in the paper:
> **[Enhancing feature compression and reconstruction in time-series and image domains with pseudo-overlap and pseudo-grouping pooling functions](https://www.sciencedirect.com/science/article/pii/S0950705126000766)** (ScienceDirect, 2026).

## ⚡ Pure PyTorch & Patch-based Efficiency

**Densatio** is built to be efficient yet accessible.
*   **No Custom CUDA Kernels:** The library is implemented entirely in **pure PyTorch**. This means it works out-of-the-box on any machine (CPU/GPU) without compilation headaches or complex C++ dependencies.
*   **Vectorized Patch Processing:** Instead of using slow Python loops, **Densatio** leverages advanced tensor reshaping (unfolding). It transforms the input into **patches**, enabling the pooling function to process the entire batch, channels, and spatial windows simultaneously using optimized matrix operations. This approach maximizes GPU throughput and avoids common bottlenecks.

## 📦 Installation

Since the package is currently under active development, you can install it directly from the source:

```bash
git clone https://github.com/matcarollo/densatio.git
cd densatio
pip install .
```

## ⚡ Multi-Dimensional Support

**Densatio** provides specialized modules for different data types:

*   **`CustomPooling1d`**: For time-series, audio, and sequential data.
*   **`CustomPooling2d`**: For images and spatial maps.
*   **`CustomPooling3d`**: For video, medical imaging (MRI/CT), and volumetric data.

## 🚀 Usage Guide

### 1. Quick Start: Built-in Methods
The easiest way to use **Densatio** is with the built-in pooling methods. You simply import the correct module for your data dimension.

**1D Example**
For sequential data with shape `(Batch, Channels, Length)`.

```python
import torch
from densatio import CustomPooling1d

# Input: (1, 64, 100)
x_1d = torch.randn(1, 64, 100)

pool_1d = CustomPooling1d(
    pool_size=2, 
    stride=2, 
    pooling_method="max_pooling"
)
out_1d = pool_1d(x_1d) # Output: (1, 64, 50)
```

**2D Example**
For spatial data with shape `(Batch, Channels, Height, Width)`.

```python
import torch
from densatio import CustomPooling2d

# Input: (1, 64, 32, 32)
x_2d = torch.randn(1, 64, 32, 32)

pool_2d = CustomPooling2d(
    pool_size=2, 
    stride=2, 
    pooling_method="max_pooling"
)
out_2d = pool_2d(x_2d) # Output: (1, 64, 16, 16)
```

**3D Example**
For volumetric data with shape `(Batch, Channels, Depth, Height, Width)`.

```python
import torch
from densatio import CustomPooling3d

# Input: (1, 64, 16, 32, 32)
x_3d = torch.randn(1, 64, 16, 32, 32)

pool_3d = CustomPooling3d(
    pool_size=2, 
    stride=2, 
    pooling_method="max_pooling"
)
out_3d = pool_3d(x_3d) # Output: (1, 64, 8, 16, 16)
```


---

### 2. Writing Your Own Custom Function
If the built-in functions aren't enough, you can write your own.

**Important:** **Densatio** preserves the spatial structure of the pooling window (it does not flatten it). This allows for complex texture analysis (like LBP), but it means your function must reduce the correct number of dimensions based on the input type.

The library provides the patch tensor ending with the channel dimension followed by the kernel dimensions:

*   **1D Input to Function:** `(..., Channels, K_l)`
    *   *Action:* Reduce the last dimension (-1).
*   **2D Input to Function:** `(..., Channels, K_h, K_w)`
    *   *Action:* Reduce the last two dimensions (-1, -2).
*   **3D Input to Function:** `(..., Channels, K_d, K_h, K_w)`
    *   *Action:* Reduce the last three dimensions (-1, -2, -3).

**Example: Recreating Max Pooling manually**

```python
import torch
from densatio import CustomPooling1d, CustomPooling2d, CustomPooling3d

# --- 1D Custom Function ---
# Reduces the last dimension (-1)
def max_pooling_1d(x):
    return torch.max(x, dim=-1, keepdim=True)[0]

max_pool_1d = CustomPooling1d(pool_size=2, stride=2, pooling_method=max_pooling_1d)

# --- 2D Custom Function ---
# Reduces the last two dimensions (-1, -2)
def max_pooling_2d(x):
    max_w = torch.max(x, dim=-1, keepdim=True)[0]
    return torch.max(max_w, dim=-2, keepdim=True)[0]

max_pool_2d = CustomPooling2d(pool_size=2, stride=2, pooling_method=max_pooling_2d)

# --- 3D Custom Function ---
# Reduces the last three dimensions (-1, -2, -3)
def max_pooling_3d(x):
    max_w = torch.max(x, dim=-1, keepdim=True)[0]
    max_h = torch.max(max_w, dim=-2, keepdim=True)[0]
    return torch.max(max_h, dim=-3, keepdim=True)[0]

max_pool_3d = CustomPooling3d(pool_size=2, stride=2, pooling_method=max_pooling_3d)
```

---

### 3. Advanced: Trainable Parameters
**Densatio** shines when you need **trainable pooling layers**. You can inject learnable parameters (scalars, vectors, or kernels) directly into the configuration.

Consider the **Weighted Max Pooling** function:
$$f(x) = \max(x^{e})$$

In **Densatio**, this is implemented in Python as follows (note the addition of a small `epsilon` for numerical stability):

```python
def weighted_max_pooling(x, e_exponent, epsilon=1e-8):
    return torch.max(torch.pow(x + epsilon, e_exponent), dim=-1, keepdim=True)[0]
```

Here `e_exponent` is a learnable parameter. You can configure how these parameters are shared across channels using the `shape` key.

#### Case A: Per-Channel Parameters (Depthwise)
Create a unique kernel for **each channel**.

```python
# Example for 1D (kernel shape is (Channels, K_l))
pool = CustomPooling1d(
    pool_size=2,
    stride=2,
    padding='same',
    pooling_method="weighted_max_pooling",
    pooling_params={
        'e_exponent': {
            # (Channels, Length) -> (3, 2)
            # 3 Channels, Kernel size 2.
            'shape': (3, 2), # Creates 3 independent kernels (one per channel).
            'init_value': 0.5,
            'requires_grad': True,
        }
    }
)
```


#### Case B: Shared Parameters (Global)
Create a single kernel shared by **all channels**.

```python
# Example for 2D (kernel shape is (1, K_h, K_w))
pool = CustomPooling2d(
    pool_size=2,
    stride=2,
    padding='same',
    pooling_method="weighted_max_pooling",
    pooling_params={
        'e_exponent': {
            # (1, Height, Width) -> (1, 2, 2)
            'shape': (1, 2, 2), # A single 2x2 kernel shared by ALL channels.
            'init_value': 0.5,
            'requires_grad': True,
        }
    }
)

```

#### Case C: Fixed Parameters (Non-Trainable)
Pass a specific tensor to initialize weights and freeze them.

```python
# Example: Fixed weights initialized from a tensor
pool = CustomPooling1d(
    pool_size=2,
    stride=2,
    pooling_method="weighted_max_pooling",
    pooling_params={
        'e_exponent': {
            # Manually setting values via Tensor.
            'init_value': torch.ones(3, 2) * 0.7, 
            'requires_grad': False # Frozen, not trainable
        }
    }
)
```

### 4. Clamped Trainable Parameters
You can optionally constrain learnable pooling parameters with `clamp=(min, max)`.

This is useful when a pooling function involves exponentiation, inverse exponents, or other operations that can become numerically unstable during training. The clamp is applied to the parameter values before calling the pooling function, while the original learnable parameter is still optimized by PyTorch.

```python
import torch
from densatio import CustomPooling2d

# Example: shared 2D exponent kernel with constrained range
pool = CustomPooling2d(
    pool_size=2,
    stride=2,
    padding='same',
    pooling_method="weighted_max_pooling_2d",
    pooling_params={
        'e_exponent': {
            'shape': (1, 2, 2),
            'init_value': 0.5,
            'requires_grad': True,
            'clamp': (1e-6, 10.0),
        }
    }
)

x = torch.rand(1, 64, 32, 32)
out = pool(x)
print(out.shape)  # (1, 64, 16, 16)
```

You can also clamp only one side of the range:

```python
'clamp': (1e-6, None)   # only lower bound
'clamp': (None, 10.0)   # only upper bound
'clamp': (None, None)   # default, no clamping
```

---

## 📚 Available Functions

### 1. Pseudo-overlap and Pseudo-grouping pooling functions (from the Paper)
These functions correspond directly to the mathematical definitions provided in the research paper Enhancing feature compression and reconstruction in time-series and image domains with pseudo-overlap and pseudo-grouping pooling functions.

| Paper Label | Function Name in Densatio | 1D | 2D | 3D |
| :--- | :--- | :---: | :---: | :---: |
| $PO_1^1$ | `product_pooling` | ✅ | ✅ | ✅ |
| $PO_1$ | `weighted_product_pooling` | ✅ | ✅ | ✅ |
| $PO_2^1$ | `min_pooling` | ✅ | ✅ | ✅ |
| $PO_2$ | `weighted_min_pooling` | ✅ | ✅ | ✅ |
| $PG_1^1$ | `max_pooling` | ✅ | ✅ | ✅ |
| $PG_1$ | `weighted_max_pooling` | ✅ | ✅ | ✅ |
| $PG_2$ | `pg_min_in` | ✅ | ✅ | ✅ |
| $PG_3$ | `pg_max_inv_exp` | ✅ | ✅ | ✅ |
| $PG_4$ | `complement_weighted_product_pooling` | ✅ | ✅ | ✅ |
| $PG_5$ | `complement_weighted_min_pooling` | ✅ | ✅ | ✅ |
| $PG_6$ | `pg_prod_in` | ✅ | ✅ | ✅ |
| $PG_7$ | `pg_prod_out_dist` | ✅ | ✅ | ✅ |

### 2. External Methods (Literature Baselines)
Implementations of other state-of-the-art pooling functions cited in the paper for comparison.

| Literature Name | Function Name in Densatio | 1D | 2D | 3D |
| :--- | :--- | :---: | :---: | :---: |
| Lip | `lip_pooling` | ✅ | ✅ | ✅ |
| SoftPool | `soft_pooling` | ✅ | ✅ | ✅ |
| Mam | `mam_pooling` | ✅ | ✅ | ✅ |
| Lbp | `lbp_pooling` | ✅ | ✅ | ✅ |
| LbpRI | `lbpri_pooling` | ✅ | ✅ | ✅ |
| Avg-topk | `avg_topk_pooling` | ✅ | ✅ | ✅ |
| T-Max-Avg | `t_max_avg_pooling` | ✅ | ✅ | ✅ |

## 📄 Citation

If you use the generalized pooling functions in your research, please cite the original paper:

```bibtex
@article{carollo2026enhancing,
  title={Enhancing feature compression and reconstruction in time-series and image domains with pseudo-overlap and pseudo-grouping pooling functions},
  author={Carollo, Matteo and Ferrero-Jaurrieta, Mikel and Paiva, Rui and Bardozzo, Francesco and Tagliaferri, Roberto},
  journal={Knowledge-Based Systems},
  pages={115333},
  year={2026},
  publisher={Elsevier}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
