import jax
import jax.numpy as jnp


from jaxsplat._types import Type


class RasterizeGaussiansFwdTypes:
    def __init__(self, num_points: int):
        self.in_xys = Type((num_points, 2), jnp.float32)
        self.in_depths = Type((num_points, 1), jnp.float32)
        self.in_radii = Type((num_points, 1), jnp.float32)
        self.in_conics = Type((num_points, 3), jnp.float32)
        self.in_num_tiles_hit = Type((num_points, 1), jnp.int32)
        self.in_colors = Type((num_points, 3), jnp.float32)
        self.in_opacity = Type((num_points, 1), jnp.float32)

        self.out_img = Type((num_points, 3), jnp.float32)
        self.out_final_Ts = Type((num_points, 1), jnp.float32)
        self.out_final_idx = Type((num_points, 1), jnp.int32)


def _rasterize_gaussians_fwd_abs(
    # input
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    num_tiles_hit: jax.Array,
    colors: jax.Array,
    opacity: jax.Array,
    # desc
    grid_dim: tuple[int, int, int],
    block_dim: tuple[int, int, int],
    img_shape: tuple[int, int, int],
):
    assert (
        xys.shape[0]
        == depths.shape[0]
        == radii.shape[0]
        == conics.shape[0]
        == num_tiles_hit.shape[0]
        == colors.shape[0]
        == opacity.shape[0]
    )
    num_points = xys.shape[0]

    t = RasterizeGaussiansFwdTypes(num_points)

    t.in_xys.assert_(xys)
    t.in_depths.assert_(depths)
    t.in_radii.assert_(radii)
    t.in_conics.assert_(conics)
    t.in_num_tiles_hit.assert_(num_tiles_hit)
    t.in_colors.assert_(colors)
    t.in_opacity.assert_(opacity)

    return (
        t.out_img.shaped_array(),
        t.out_final_Ts.shaped_array(),
        t.out_final_idx.shaped_array(),
    )
