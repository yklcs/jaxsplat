from jax.interpreters import mlir
from jax.interpreters.mlir import ir
import jax.numpy as jnp
from jax.typing import DTypeLike


def layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


class Type:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    ir_type: ir.Type
    ir_tensor_type: ir.RankedTensorType
    layout: tuple

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike):
        self.shape = shape
        self.dtype = jnp.dtype(dtype)
        self.ir_type = mlir.dtype_to_ir_type(self.dtype)
        self.ir_tensor_type = ir.RankedTensorType.get(shape, self.ir_type)
        self.layout = layout(shape)
