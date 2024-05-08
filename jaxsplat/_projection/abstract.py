import jax
from jax.core import ShapedArray
from jax.dtypes import canonicalize_dtype
import jax.numpy as jnp


def _project_gaussians_fwd_abs(
    # input
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    # desc
    num_points: int,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int, int],
    block_width: int,
    clip_thresh: float,
):
    assert means3d.shape[0] == scales.shape[0] == quats.shape[0]
    num_points = means3d.shape[0]

    assert means3d.shape == (num_points, 3)
    assert scales.shape == (num_points, 3)
    assert quats.shape == (num_points, 4)
    assert viewmat.shape == (4, 4)

    assert canonicalize_dtype(means3d.dtype) == jnp.float32
    assert canonicalize_dtype(scales.dtype) == jnp.float32
    assert canonicalize_dtype(quats.dtype) == jnp.float32
    assert canonicalize_dtype(viewmat.dtype) == jnp.float32

    out_covs3d = ShapedArray((num_points, 3), jnp.float32)
    out_xys = ShapedArray((num_points, 2), jnp.float32)
    out_depths = ShapedArray((num_points, 1), jnp.float32)
    out_radii = ShapedArray((num_points, 1), jnp.int32)
    out_conics = ShapedArray((num_points, 3), jnp.float32)
    out_compensation = ShapedArray((num_points, 1), jnp.float32)
    out_num_tiles_hit = ShapedArray((num_points, 1), jnp.uint32)

    return (
        out_covs3d,
        out_xys,
        out_depths,
        out_radii,
        out_conics,
        out_compensation,
        out_num_tiles_hit,
    )
