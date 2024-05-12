import jax
import jax.numpy as jnp

from jaxsplat._types import Type


class RasterizeFwdTypes:
    def __init__(
        self,
        num_points: int,
        num_intersects: int,
        img_shape: tuple[int, int],
        block_width: int,
    ):
        num_tiles = ((img_shape[0] + block_width - 1) // block_width) * (
            (img_shape[1] + block_width - 1) // block_width
        )

        self.in_colors = Type((num_points, 3), jnp.float32)
        self.in_opacities = Type((num_points, 1), jnp.float32)
        self.in_background = Type((3,), jnp.float32)
        self.in_xys = Type((num_points, 2), jnp.float32)
        self.in_depths = Type((num_points, 1), jnp.float32)
        self.in_radii = Type((num_points, 1), jnp.int32)
        self.in_conics = Type((num_points, 3), jnp.float32)
        self.in_cum_tiles_hit = Type((num_points, 1), jnp.uint32)

        self.out_gaussian_ids_sorted = Type((num_intersects, 1), jnp.int32)
        self.out_tile_bins = Type((num_tiles, 2), jnp.int32)
        self.out_final_Ts = Type((*img_shape, 1), jnp.float32)
        self.out_final_idx = Type((*img_shape, 1), jnp.int32)
        self.out_img = Type((*img_shape, 3), jnp.float32)


def _rasterize_fwd_abs(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    #
    num_points: int,
    num_intersects: int,
    img_shape: tuple[int, int],
    block_width: int,
):
    t = RasterizeFwdTypes(
        num_points,
        num_intersects,
        img_shape,
        block_width,
    )

    t.in_colors.assert_(colors)
    t.in_opacities.assert_(opacities)
    t.in_background.assert_(background)
    t.in_xys.assert_(xys)
    t.in_depths.assert_(depths)
    t.in_radii.assert_(radii)
    t.in_conics.assert_(conics)
    t.in_cum_tiles_hit.assert_(cum_tiles_hit)

    return (
        t.out_gaussian_ids_sorted.shaped_array(),
        t.out_tile_bins.shaped_array(),
        t.out_final_Ts.shaped_array(),
        t.out_final_idx.shaped_array(),
        t.out_img.shaped_array(),
    )


class RasterizeBwdTypes:
    def __init__(
        self,
        num_points: int,
        num_intersects: int,
        img_shape: tuple[int, int],
        block_width: int,
    ):
        num_tiles = ((img_shape[0] + block_width - 1) // block_width) * (
            (img_shape[1] + block_width - 1) // block_width
        )
        self.in_colors = Type((num_points, 3), jnp.float32)
        self.in_opacities = Type((num_points, 1), jnp.float32)
        self.in_background = Type((3,), jnp.float32)
        self.in_xys = Type((num_points, 2), jnp.float32)
        self.in_conics = Type((num_points, 3), jnp.float32)
        self.in_gaussian_ids_sorted = Type((num_intersects, 1), jnp.int32)
        self.in_tile_bins = Type((num_tiles, 2), jnp.int32)
        self.in_final_Ts = Type((*img_shape, 1), jnp.float32)
        self.in_final_idx = Type((*img_shape, 1), jnp.int32)
        self.in_v_img = Type((*img_shape, 3), jnp.float32)
        self.in_v_img_alpha = Type((*img_shape, 1), jnp.float32)

        self.out_v_color = Type((num_points, 3), jnp.float32)
        self.out_v_opacity = Type((num_points, 1), jnp.float32)
        self.out_v_xy = Type((num_points, 2), jnp.float32)
        self.out_v_xy_abs = Type((num_points, 2), jnp.float32)
        self.out_v_conic = Type((num_points, 3), jnp.float32)


def _rasterize_bwd_abs(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    conics: jax.Array,
    gaussian_ids_sorted: jax.Array,
    tile_bins: jax.Array,
    final_Ts: jax.Array,
    final_idx: jax.Array,
    v_img: jax.Array,
    v_img_alpha: jax.Array,
    #
    num_points: int,
    num_intersects: int,
    img_shape: tuple[int, int],
    block_width: int,
):
    t = RasterizeBwdTypes(num_points, num_intersects, img_shape, block_width)

    t.in_colors.assert_(colors)
    t.in_opacities.assert_(opacities)
    t.in_background.assert_(background)
    t.in_xys.assert_(xys)
    t.in_conics.assert_(conics)
    t.in_gaussian_ids_sorted.assert_(gaussian_ids_sorted)
    t.in_tile_bins.assert_(tile_bins)
    t.in_final_Ts.assert_(final_Ts)
    t.in_final_idx.assert_(final_idx)
    t.in_v_img.assert_(v_img)
    t.in_v_img_alpha.assert_(v_img_alpha)

    return (
        t.out_v_color.shaped_array(),
        t.out_v_opacity.shaped_array(),
        t.out_v_xy.shaped_array(),
        t.out_v_xy_abs.shaped_array(),
        t.out_v_conic.shaped_array(),
    )
