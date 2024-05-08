import jax

from jaxsplat._projection import impl


def project_gaussians_fwd(
    # input
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    # desc
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[float, float],
    block_width: int,
    clip_thresh: float,
):
    num_points = means3d.shape[0]

    return impl._project_gaussians_fwd_p.bind(
        means3d,
        scales,
        quats,
        viewmat,
        # desc
        num_points=num_points,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )
