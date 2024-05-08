import jax
import jax.numpy as jnp

from jaxsplat._types import Type


class ProjectGaussiansFwdTypes:
    def __init__(self, num_points: int):
        self.in_means3d = Type((num_points, 3), jnp.float32)
        self.in_scales = Type((num_points, 3), jnp.float32)
        self.in_quats = Type((num_points, 4), jnp.float32)
        self.in_viewmat = Type((4, 4), jnp.float32)

        self.out_covs3d = Type((num_points, 3), jnp.float32)
        self.out_xys = Type((num_points, 2), jnp.float32)
        self.out_depths = Type((num_points, 1), jnp.float32)
        self.out_radii = Type((num_points, 1), jnp.int32)
        self.out_conics = Type((num_points, 3), jnp.float32)
        self.out_compensation = Type((num_points, 1), jnp.float32)
        self.out_num_tiles_hit = Type((num_points, 1), jnp.uint32)


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

    t = ProjectGaussiansFwdTypes(num_points)

    t.in_means3d.assert_(means3d)
    t.in_scales.assert_(scales)
    t.in_quats.assert_(quats)
    t.in_viewmat.assert_(viewmat)

    return (
        t.out_covs3d.shaped_array(),
        t.out_xys.shaped_array(),
        t.out_depths.shaped_array(),
        t.out_radii.shaped_array(),
        t.out_conics.shaped_array(),
        t.out_compensation.shaped_array(),
        t.out_num_tiles_hit.shaped_array(),
    )
