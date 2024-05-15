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
    background: jax.Array,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[int, int],
    glob_scale: float,
    clip_thresh: float,
    block_size: int = 16,
) -> jax.Array:
    """
    Renders 3D Gaussians to a 2D image differentiably.
    Output is differentiable w.r.t. all non-keyword-only arguments.

    Args:
        means3d (Array): (N, 3) array of 3D Gaussian means
        scales (Array): (N, 3) array of 3D Gaussian scales
        quats (Array): (N, 4) array of 3D Gaussian quaternions, must be normalized
        colors (Array): (N, 3) array of 3D Gaussian colors
        opacities (Array): (N, 1) array of 3D Gaussian opacities
    Keyword Args:
        viewmat (Array): (4, 4) array containing view matrix
        background (Array): (3,) array of background color
        img_shape (tuple[int, int]): Image shape in (H, W)
        f (tuple[float, float]): Focal lengths in (fx, fy)
        c (tuple[int, int]): Principal points in (cx, cy)
        glob_scale (float): Global scaling factor
        clip_thresh (float): Minimum z depth clipping threshold
        block_size (int): CUDA block size, 1 < block_size <= 16.
    Returns:
        Array: Rendered image
    """
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
