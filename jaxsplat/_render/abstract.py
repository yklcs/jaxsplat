import jax
import jax.numpy as jnp


from jaxsplat._types import Type


class RenderFwdTypes:
    def __init__(self, num_points: int):
        self.in_mean3ds = Type((num_points, 3), jnp.float32)
        self.in_scales = Type((num_points, 3), jnp.float32)
        self.in_quats = Type((num_points, 4), jnp.float32)
        self.in_viewmat = Type((4, 4), jnp.float32)
        self.in_colors = Type((num_points, 3), jnp.float32)
        self.in_opacities = Type((num_points, 1), jnp.float32)
        self.in_background = Type((3,), jnp.float32)

        self.out_cov3ds = Type((num_points, 3), jnp.float32)
        self.out_xys = Type((num_points, 2), jnp.float32)
        self.out_depths = Type((num_points, 1), jnp.float32)
        self.out_radii = Type((num_points, 1), jnp.int32)
        self.out_conics = Type((num_points, 3), jnp.float32)
        self.out_compensation = Type((num_points, 1), jnp.float32)
        self.out_num_tiles_hit = Type((num_points, 1), jnp.uint32)
        self.out_gaussian_ids_sorted = Type((num_points, 1), jnp.int32)
        self.out_tile_bins = Type((num_points, 2), jnp.int32)
        self.out_out_img = Type((num_points, 3), jnp.float32)
        self.out_final_Ts = Type((num_points, 1), jnp.float32)
        self.out_final_idx = Type((num_points, 1), jnp.int32)


def _render_fwd_abs(
    # input
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    # desc
    num_points: int,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int, int],
    block_width: int,
    clip_thresh: float,
):
    assert (
        mean3ds.shape[0]
        == scales.shape[0]
        == quats.shape[0]
        == colors.shape[0]
        == opacities.shape[0]
    )
    num_points = mean3ds.shape[0]

    t = RenderFwdTypes(num_points)

    t.in_mean3ds.assert_(mean3ds)
    t.in_scales.assert_(scales)
    t.in_quats.assert_(quats)
    t.in_viewmat.assert_(viewmat)
    t.in_colors.assert_(colors)
    t.in_opacities.assert_(opacities)
    t.in_background.assert_(background)

    return (
        t.out_cov3ds.shaped_array(),
        t.out_xys.shaped_array(),
        t.out_depths.shaped_array(),
        t.out_radii.shaped_array(),
        t.out_conics.shaped_array(),
        t.out_compensation.shaped_array(),
        t.out_num_tiles_hit.shaped_array(),
        t.out_gaussian_ids_sorted.shaped_array(),
        t.out_tile_bins.shaped_array(),
        t.out_out_img.shaped_array(),
        t.out_final_Ts.shaped_array(),
        t.out_final_idx.shaped_array(),
    )


class RenderBwdTypes:
    def __init__(self, num_points: int):
        self.in_mean3ds = Type((num_points, 3), jnp.float32)
        self.in_scales = Type((num_points, 3), jnp.float32)
        self.in_quats = Type((num_points, 4), jnp.float32)
        self.in_viewmat = Type((4, 4), jnp.float32)
        self.in_colors = Type((num_points, 3), jnp.float32)
        self.in_opacities = Type((num_points, 1), jnp.float32)
        self.in_background = Type((3,), jnp.float32)
        self.in_num_tiles_hit = Type((num_points, 1), jnp.uint32)
        self.in_gaussian_ids_sorted = Type((num_points, 1), jnp.int32)
        self.in_tile_bins = Type((num_points, 2), jnp.int32)
        self.in_final_Ts = Type((num_points, 1), jnp.float32)
        self.in_final_idx = Type((num_points, 1), jnp.int32)
        self.in_v_out_img = Type((num_points, 3), jnp.float32)
        self.in_v_out_img_alpha = Type((num_points, 1), jnp.float32)
        self.in_v_compensation = Type((num_points, 1), jnp.float32)

        self.out_v_mean3d = Type((num_points, 3), jnp.float32)
        self.out_v_scale = Type((num_points, 3), jnp.float32)
        self.out_v_quat = Type((num_points, 4), jnp.float32)
        self.out_color = Type((num_points, 3), jnp.float32)
        self.out_opacity = Type((num_points, 1), jnp.float32)
        self.out_v_cov2d = Type((num_points, 3), jnp.float32)
        self.out_v_xy = Type((num_points, 2), jnp.float32)
        self.out_v_xy_abs = Type((num_points, 2), jnp.float32)
        self.out_v_depth = Type((num_points, 1), jnp.float32)
        self.out_v_conic = Type((num_points, 3), jnp.float32)
        self.out_v_cov3d = Type((num_points, 1), jnp.float32)


def _render_bwd_abs(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    num_tiles_hit: jax.Array,
    gaussian_ids_sorted: jax.Array,
    tile_bins: jax.Array,
    final_Ts: jax.Array,
    final_idx: jax.Array,
    v_out_img: jax.Array,
    v_out_img_alpha: jax.Array,
    v_compensation: jax.Array,
    num_points: int,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int, int],
):
    assert (
        mean3ds.shape[0]
        == scales.shape[0]
        == quats.shape[0]
        == colors.shape[0]
        == opacities.shape[0]
        == num_tiles_hit.shape[0]
        == gaussian_ids_sorted.shape[0]
        == tile_bins.shape[0]
        == final_Ts.shape[0]
        == final_idx.shape[0]
        == v_out_img.shape[0]
        == v_out_img_alpha.shape[0]
        == v_compensation.shape[0]
    )
    num_points = mean3ds.shape[0]

    t = RenderBwdTypes(num_points)

    t.in_mean3ds.assert_(mean3ds)
    t.in_scales.assert_(scales)
    t.in_quats.assert_(quats)
    t.in_viewmat.assert_(viewmat)
    t.in_colors.assert_(colors)
    t.in_opacities.assert_(opacities)
    t.in_background.assert_(background)
    t.in_num_tiles_hit.assert_(num_tiles_hit)
    t.in_gaussian_ids_sorted.assert_(gaussian_ids_sorted)
    t.in_tile_bins.assert_(tile_bins)
    t.in_final_Ts.assert_(final_Ts)
    t.in_final_idx.assert_(final_idx)
    t.in_v_out_img.assert_(v_out_img)
    t.in_v_out_img_alpha.assert_(v_out_img_alpha)
    t.in_v_compensation.assert_(v_compensation)

    return (
        t.out_v_mean3d.shaped_array(),
        t.out_v_scale.shaped_array(),
        t.out_v_quat.shaped_array(),
        t.out_color.shaped_array(),
        t.out_opacity.shaped_array(),
        t.out_v_cov2d.shaped_array(),
        t.out_v_xy.shaped_array(),
        t.out_v_xy_abs.shaped_array(),
        t.out_v_depth.shaped_array(),
        t.out_v_conic.shaped_array(),
        t.out_v_cov3d.shaped_array(),
    )
