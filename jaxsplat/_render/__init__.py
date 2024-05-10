import jax

from typing import TypedDict

from jaxsplat._render import impl


def render(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    *,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int, int],
    block_width: int,
    clip_thresh: float,
) -> jax.Array:
    (out_img, _out_img_alpha, _compensation) = _render(
        mean3ds,
        scales,
        quats,
        viewmat,
        colors,
        opacities,
        background,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )
    return out_img


class Residuals(TypedDict):
    mean3ds: jax.Array
    scales: jax.Array
    quats: jax.Array
    viewmat: jax.Array
    colors: jax.Array
    opacities: jax.Array
    background: jax.Array
    num_tiles_hit: jax.Array
    gaussian_ids_sorted: jax.Array
    tile_bins: jax.Array
    final_Ts: jax.Array
    final_idx: jax.Array
    num_points: int
    glob_scale: float
    f: tuple[float, float]
    c: tuple[float, float]
    img_shape: tuple[int, int]


@jax.custom_vjp
def _render(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int, int],
    block_width: int,
    clip_thresh: float,
):
    primals, _ = _render_fwd(
        mean3ds,
        scales,
        quats,
        viewmat,
        colors,
        opacities,
        background,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )

    return primals


def _render_fwd(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int, int],
    block_width: int,
    clip_thresh: float,
):
    num_points = mean3ds.shape[0]

    (
        cov3ds,
        xys,
        depths,
        radii,
        conics,
        compensation,
        num_tiles_hit,
        gaussian_ids_sorted,
        tile_bins,
        out_img,
        final_Ts,
        final_idx,
    ) = impl._render_fwd_p.bind(
        mean3ds,
        scales,
        quats,
        viewmat,
        colors,
        opacities,
        background,
        num_points=num_points,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )

    out_img_alpha = 1 - final_Ts
    primals = (
        out_img,
        out_img_alpha,
        compensation,
    )

    residuals: Residuals = {
        "mean3ds": mean3ds,
        "scales": scales,
        "quats": quats,
        "viewmat": viewmat,
        "colors": colors,
        "opacities": opacities,
        "background": background,
        "num_tiles_hit": num_tiles_hit,
        "gaussian_ids_sorted": gaussian_ids_sorted,
        "tile_bins": tile_bins,
        "final_Ts": final_Ts,
        "final_idx": final_idx,
        "num_points": num_points,
        "glob_scale": glob_scale,
        "f": f,
        "c": c,
        "img_shape": img_shape,
    }

    return primals, residuals


def _render_bwd(
    residuals,
    cotangents,
):
    (
        v_out_img,
        v_out_img_alpha,
        v_compensation,
    ) = cotangents

    num_points = residuals.mean3ds.shape[0]

    (
        v_mean3d,
        v_scale,
        v_quat,
        v_color,
        v_opacity,
        v_cov2d,
        v_xy,
        v_xy_abs,
        v_depth,
        v_conic,
        v_cov3d,
    ) = impl._render_bwd_p.bind(
        residuals.mean3ds,
        residuals.scales,
        residuals.quats,
        residuals.viewmat,
        residuals.colors,
        residuals.opacities,
        residuals.background,
        residuals.num_tiles_hit,
        residuals.gaussian_ids_sorted,
        residuals.tile_bins,
        residuals.final_Ts,
        residuals.final_idx,
        v_out_img,
        v_out_img_alpha,
        v_compensation,
        num_points=num_points,
        glob_scale=residuals.glob_scale,
        f=residuals.f,
        c=residuals.c,
        img_shape=residuals.img_shape,
    )

    return (
        v_mean3d,
        v_scale,
        v_quat,
        None,
        v_color,
        v_opacity,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_render.defvjp(_render_fwd, _render_bwd)

__all__ = ["render"]
