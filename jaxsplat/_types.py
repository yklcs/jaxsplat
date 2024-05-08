import jax
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.core import ShapedArray, canonicalize_shape
import jax.numpy as jnp
from jax.typing import DTypeLike
from jax.dtypes import canonicalize_dtype


class Type:
    shape: tuple[int, ...]
    dtype: jnp.dtype

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike):
        self.shape = canonicalize_shape(shape)
        self.dtype = jnp.dtype(dtype)

    def ir_type(self):
        return mlir.dtype_to_ir_type(self.dtype)

    def ir_tensor_type(self):
        return ir.RankedTensorType.get(self.shape, self.ir_type())

    def layout(self):
        return tuple(range(len(self.shape) - 1, -1, -1))

    def shaped_array(self):
        return ShapedArray(self.shape, self.dtype)

    def assert_(self, other: jax.Array):
        assert self.shape == other.shape and canonicalize_dtype(
            self.dtype
        ) == canonicalize_dtype(other.dtype)
