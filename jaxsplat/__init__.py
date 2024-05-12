import jax
from jaxsplat._project import project
from jaxsplat._rasterize import rasterize


def render(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    *,
    viewmat: jax.Array,
    background: jax.Array = jax.numpy.ones((1,)),
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[int, int],
    glob_scale: float,
    clip_thresh: float,
    block_size: int = 16,
):
    (xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit) = project(
        means3d,
        scales,
        quats,
        viewmat,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
        block_width=block_size,
    )

    img = rasterize(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        img_shape=img_shape,
        block_width=block_size,
    )

    return img


__all__ = ["render", "project", "rasterize"]
