import jax

from jaxsplat._rasterization import impl


def rasterize_gaussians(
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
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]: ...


# @jax.custom_vjp
def _rasterize_gaussians(
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    num_tiles_hit: jax.Array,
    colors: jax.Array,
    opacity: jax.Array,
    img_shape: tuple[int, int, int],
): ...


def _rasterize_gaussians_fwd(
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    num_tiles_hit: jax.Array,
    colors: jax.Array,
    opacity: jax.Array,
    img_shape: tuple[int, int, int],
    block_width: int,
): ...


__all__ = ["rasterize_gaussians"]
