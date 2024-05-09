import jax

from jaxsplat._projection import impl


def project_gaussians(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[float, float],
    block_width: int,
    clip_thresh: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return _project_gaussians(
        means3d,
        scales,
        quats,
        viewmat,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )


@jax.custom_vjp
def _project_gaussians(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[float, float],
    block_width: int,
    clip_thresh: float,
):
    primals, _ = _project_gaussians_fwd(
        means3d,
        scales,
        quats,
        viewmat,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )

    return primals


def _project_gaussians_fwd(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[float, float],
    block_width: int,
    clip_thresh: float,
):
    num_points = means3d.shape[0]

    (
        covs3d,
        xys,
        depths,
        radii,
        conics,
        compensation,
        num_tiles_hit,
    ) = impl._project_gaussians_fwd_p.bind(
        means3d,
        scales,
        quats,
        viewmat,
        num_points=num_points,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )

    primals = (
        covs3d,
        xys,
        depths,
        radii,
        conics,
        compensation,
        num_tiles_hit,
    )

    residuals = (
        means3d,
        scales,
        quats,
        viewmat,
        covs3d,
        radii,
        conics,
        compensation,
        glob_scale,
        f,
        c,
        img_shape,
    )

    return primals, residuals


def _project_gaussians_bwd(
    residuals,
    cotangents,
):
    (
        means3d,
        scales,
        quats,
        viewmat,
        cov3d,
        radii,
        conics,
        compensation,
        glob_scale,
        f,
        c,
        img_shape,
    ) = residuals

    (
        _v_covs3d,
        v_xy,
        v_depth,
        _v_radii,
        v_conic,
        v_compensation,
        _v_num_tiles_hit,
    ) = cotangents

    num_points = means3d.shape[0]

    (
        _v_cov2d,
        _v_cov3d,
        v_mean3d,
        v_scale,
        v_quat,
    ) = impl._project_gaussians_bwd_p.bind(
        means3d,
        scales,
        quats,
        viewmat,
        cov3d,
        radii,
        conics,
        compensation,
        v_xy,
        v_depth,
        v_conic,
        v_compensation,
        num_points=num_points,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
    )

    return (
        v_mean3d,
        v_scale,
        v_quat,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_project_gaussians.defvjp(_project_gaussians_fwd, _project_gaussians_bwd)

__all__ = ["project_gaussians"]
