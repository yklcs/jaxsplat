import jax
import jax.numpy as jnp

from jaxsplat._types import Type


class ProjectFwdTypes:
    def __init__(self, num_points: int):
        self.in_mean3ds = Type((num_points, 3), jnp.float32)
        self.in_scales = Type((num_points, 3), jnp.float32)
        self.in_quats = Type((num_points, 4), jnp.float32)
        self.in_viewmat = Type((4, 4), jnp.float32)

        self.out_cov3ds = Type((num_points, 6), jnp.float32)
        self.out_xys = Type((num_points, 2), jnp.float32)
        self.out_depths = Type((num_points, 1), jnp.float32)
        self.out_radii = Type((num_points, 1), jnp.int32)
        self.out_conics = Type((num_points, 3), jnp.float32)
        self.out_compensation = Type((num_points, 1), jnp.float32)
        self.out_num_tiles_hit = Type((num_points, 1), jnp.uint32)
        self.out_cum_tiles_hit = Type((num_points, 1), jnp.uint32)


def _project_fwd_abs(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    #
    num_points: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    t = ProjectFwdTypes(num_points)

    t.in_mean3ds.assert_(mean3ds)
    t.in_scales.assert_(scales)
    t.in_quats.assert_(quats)
    t.in_viewmat.assert_(viewmat)

    return (
        t.out_cov3ds.shaped_array(),
        t.out_xys.shaped_array(),
        t.out_depths.shaped_array(),
        t.out_radii.shaped_array(),
        t.out_conics.shaped_array(),
        t.out_compensation.shaped_array(),
        t.out_num_tiles_hit.shaped_array(),
        t.out_cum_tiles_hit.shaped_array(),
    )


class ProjectBwdTypes:
    def __init__(self, num_points: int):
        self.in_mean3ds = Type((num_points, 3), jnp.float32)
        self.in_scales = Type((num_points, 3), jnp.float32)
        self.in_quats = Type((num_points, 4), jnp.float32)
        self.in_viewmat = Type((4, 4), jnp.float32)
        self.in_cov3ds = Type((num_points, 6), jnp.float32)
        self.in_xys = Type((num_points, 2), jnp.float32)
        self.in_radii = Type((num_points, 1), jnp.int32)
        self.in_conics = Type((num_points, 3), jnp.float32)
        self.in_compensation = Type((num_points, 1), jnp.float32)
        self.in_v_compensation = Type((num_points, 1), jnp.float32)
        self.in_v_xy = Type((num_points, 2), jnp.float32)
        self.in_v_depth = Type((num_points, 1), jnp.float32)
        self.in_v_conic = Type((num_points, 3), jnp.float32)

        self.out_v_mean3d = Type((num_points, 3), jnp.float32)
        self.out_v_scale = Type((num_points, 3), jnp.float32)
        self.out_v_quat = Type((num_points, 4), jnp.float32)
        self.out_v_cov2d = Type((num_points, 3), jnp.float32)
        self.out_v_cov3d = Type((num_points, 1), jnp.float32)


def _project_bwd_abs(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    cov3ds: jax.Array,
    xys: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    compensation: jax.Array,
    v_compensation: jax.Array,
    v_xy: jax.Array,
    v_depth: jax.Array,
    v_conic: jax.Array,
    #
    num_points: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    t = ProjectBwdTypes(num_points)

    t.in_mean3ds.assert_(mean3ds)
    t.in_scales.assert_(scales)
    t.in_quats.assert_(quats)
    t.in_viewmat.assert_(viewmat)
    t.in_cov3ds.assert_(cov3ds)
    t.in_xys.assert_(xys)
    t.in_radii.assert_(radii)
    t.in_conics.assert_(conics)
    t.in_compensation.assert_(compensation)
    t.in_v_compensation.assert_(v_compensation)
    t.in_v_xy.assert_(v_xy)
    t.in_v_depth.assert_(v_depth)
    t.in_v_conic.assert_(v_conic)

    return (
        t.out_v_mean3d.shaped_array(),
        t.out_v_scale.shaped_array(),
        t.out_v_quat.shaped_array(),
        t.out_v_cov2d.shaped_array(),
        t.out_v_cov3d.shaped_array(),
    )
