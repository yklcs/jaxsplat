"""jaxsplat on the XLA FFI (works with JAX >= 0.6).

Drop-in replacement for ``jaxsplat``: ``render``, ``project`` and
``rasterize`` take the same arguments and return the same values, so
``import jaxsplat_ffi as jaxsplat`` is enough to switch.

Differences to the original:
  * Works with current JAX (no ``register_custom_call_target``).
  * ``vmap`` renders the whole batch in one launch per kernel (no custom
    batching rules needed).
  * Gaussians with opacity < 1/255 are skipped before sorting (same image,
    much faster when many Gaussians are invisible).
  * ``background`` also receives a gradient.
  * Principal point ``c`` is kept as float (the original rounds through int
    in some callers).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxsplat_ffi import _ffi

__all__ = ["render", "project", "rasterize"]

_F32 = jnp.float32
_I32 = jnp.int32


# Batched calls: JAX broadcasts all arguments to the batch shape and the CUDA
# side renders the whole batch in one launch per kernel.
_VMAP = "broadcast_all"


def _sds(shape, dtype=_F32):
    return jax.ShapeDtypeStruct(shape, dtype)


def _check_block_width(block_width: int) -> int:
    block_width = int(block_width)
    if not 2 <= block_width <= 16:
        raise ValueError(f"block_size must be in [2, 16], got {block_width}")
    return block_width


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProjectConfig:
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float
    block_width: int

    def camera_attrs(self) -> dict:
        return dict(
            glob_scale=np.float32(self.glob_scale),
            fx=np.float32(self.f[0]),
            fy=np.float32(self.f[1]),
            cx=np.float32(self.c[0]),
            cy=np.float32(self.c[1]),
            img_h=np.int32(self.img_shape[0]),
            img_w=np.int32(self.img_shape[1]),
        )


def _project_fwd_call(cfg: _ProjectConfig, means3d, scales, quats, viewmat, opacities):
    n = means3d.shape[0]
    out_types = (
        _sds((n, 6)),  # cov3ds
        _sds((n, 2)),  # xys
        _sds((n, 1)),  # depths
        _sds((n, 1), _I32),  # radii
        _sds((n, 3)),  # conics
        _sds((n, 1)),  # compensation
        _sds((n, 1), _I32),  # num_tiles_hit
        _sds((n, 1), _I32),  # cum_tiles_hit
    )
    return jax.ffi.ffi_call(
        "jaxsplat_ffi_project_fwd", out_types, vmap_method=_VMAP
    )(
        means3d,
        scales,
        quats,
        viewmat,
        opacities,
        **cfg.camera_attrs(),
        block_width=np.int32(cfg.block_width),
        clip_thresh=np.float32(cfg.clip_thresh),
    )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _project(cfg, means3d, scales, quats, viewmat, opacities):
    return _project_vjp_fwd(cfg, means3d, scales, quats, viewmat, opacities)[0]


def _project_vjp_fwd(cfg, means3d, scales, quats, viewmat, opacities):
    # opacities are only used to skip invisible Gaussians (opacity < 1/255);
    # that does not change the image, so they get no gradient here.
    (cov3ds, xys, depths, radii, conics, compensation, num_tiles_hit, cum_tiles_hit) = (
        _project_fwd_call(cfg, means3d, scales, quats, viewmat, opacities)
    )
    primals = (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit, compensation)
    residuals = (means3d, scales, quats, viewmat, opacities, cov3ds, radii, conics, compensation)
    return primals, residuals


def _project_vjp_bwd(cfg, residuals, cotangents):
    means3d, scales, quats, viewmat, opacities, cov3ds, radii, conics, compensation = residuals
    v_xy, v_depth, _, v_conic, _, _, v_compensation = cotangents
    n = means3d.shape[0]
    _v_cov2d, _v_cov3d, v_mean3d, v_scale, v_quat = jax.ffi.ffi_call(
        "jaxsplat_ffi_project_bwd",
        (_sds((n, 3)), _sds((n, 6)), _sds((n, 3)), _sds((n, 3)), _sds((n, 4))),
        vmap_method=_VMAP,
    )(
        means3d,
        scales,
        quats,
        viewmat,
        cov3ds,
        radii,
        conics,
        compensation,
        v_xy,
        v_depth,
        v_conic,
        v_compensation,
        **cfg.camera_attrs(),
    )
    # No gradient w.r.t. the view matrix (same as the original jaxsplat).
    return v_mean3d, v_scale, v_quat, jnp.zeros_like(viewmat), jnp.zeros_like(opacities)


_project.defvjp(_project_vjp_fwd, _project_vjp_bwd)


def project(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int = 16,
    opacities: jax.Array | None = None,
):
    """Projects 3D Gaussians to 2D.

    Returns ``(xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit)``
    exactly like ``jaxsplat.project``. If ``opacities`` (N, 1) is given,
    Gaussians with opacity < 1/255 are skipped (radius 0); they would not be
    drawn anyway, so the rendered image is unchanged.
    """
    _ffi.ensure_loaded()
    cfg = _ProjectConfig(
        img_shape=(int(img_shape[0]), int(img_shape[1])),
        f=(float(f[0]), float(f[1])),
        c=(float(c[0]), float(c[1])),
        glob_scale=float(glob_scale),
        clip_thresh=float(clip_thresh),
        block_width=_check_block_width(block_width),
    )
    means3d = jnp.asarray(means3d, _F32)
    n = means3d.shape[0]
    if opacities is None:
        opacities = jnp.ones((n, 1), _F32)
    out = _project(
        cfg,
        means3d,
        jnp.asarray(scales, _F32),
        jnp.asarray(quats, _F32),
        jnp.asarray(viewmat, _F32),
        jnp.asarray(opacities, _F32).reshape(n, 1),
    )
    return out[:6]


# ---------------------------------------------------------------------------
# rasterize
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RasterizeConfig:
    img_shape: tuple[int, int]
    block_width: int

    def attrs(self) -> dict:
        return dict(
            img_h=np.int32(self.img_shape[0]),
            img_w=np.int32(self.img_shape[1]),
            block_width=np.int32(self.block_width),
        )


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _rasterize(cfg, colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit):
    return _rasterize_vjp_fwd(
        cfg, colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit
    )[0]


def _rasterize_vjp_fwd(cfg, colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit):
    h, w = cfg.img_shape
    final_Ts, final_idx, img = jax.ffi.ffi_call(
        "jaxsplat_ffi_rasterize_fwd",
        (_sds((h, w, 1)), _sds((h, w, 1), _I32), _sds((h, w, 3))),
        vmap_method=_VMAP,
    )(colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit, **cfg.attrs())
    img_alpha = 1.0 - final_Ts
    residuals = (
        colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit,
        final_Ts, final_idx,
    )
    return (img, img_alpha), residuals


def _rasterize_vjp_bwd(cfg, residuals, cotangents):
    (colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit,
     final_Ts, final_idx) = residuals
    v_img, v_img_alpha = cotangents
    n = colors.shape[0]
    v_colors, v_opacity, v_xy, v_conic = jax.ffi.ffi_call(
        "jaxsplat_ffi_rasterize_bwd",
        (_sds((n, 3)), _sds((n, 1)), _sds((n, 2)), _sds((n, 3))),
        vmap_method=_VMAP,
    )(
        colors, opacities, background, xys, depths, radii, conics, cum_tiles_hit,
        final_Ts, final_idx, v_img, v_img_alpha,
        **cfg.attrs(),
    )
    # img = rendered + T_final * background  =>  d img / d background = T_final
    v_background = jnp.sum(v_img * final_Ts, axis=(0, 1))
    return (
        v_colors,
        v_opacity,
        v_background,
        v_xy,
        jnp.zeros_like(depths),
        None,  # radii (int)
        v_conic,
        None,  # cum_tiles_hit (int)
    )


_rasterize.defvjp(_rasterize_vjp_fwd, _rasterize_vjp_bwd)


def rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    *,
    img_shape: tuple[int, int],
    block_width: int = 16,
) -> jax.Array:
    """Rasterizes projected Gaussians. Returns the (H, W, 3) image."""
    _ffi.ensure_loaded()
    cfg = _RasterizeConfig(
        img_shape=(int(img_shape[0]), int(img_shape[1])),
        block_width=_check_block_width(block_width),
    )
    n = colors.shape[0]
    img, _img_alpha = _rasterize(
        cfg,
        jnp.asarray(colors, _F32),
        jnp.asarray(opacities, _F32).reshape(n, 1),
        jnp.asarray(background, _F32).reshape(3),
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
    )
    return img


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


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
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_size: int = 16,
) -> jax.Array:
    """Renders 3D Gaussians to an (H, W, 3) image, differentiably.

    Same signature as ``jaxsplat.render``. Differentiable w.r.t. means3d,
    scales, quats, colors, opacities and background.

    Args:
        means3d: (N, 3) Gaussian centres.
        scales: (N, 3) Gaussian scales.
        quats: (N, 4) quaternions (w, x, y, z), must be normalised.
        colors: (N, 3) colours.
        opacities: (N, 1) opacities.
        viewmat: (4, 4) world-to-camera matrix.
        background: (3,) background colour.
        img_shape: (H, W).
        f: focal lengths (fx, fy).
        c: principal point (cx, cy).
        glob_scale: global scale factor for the Gaussians.
        clip_thresh: near-plane depth threshold.
        block_size: tile size, 2 <= block_size <= 16.
    """
    xys, depths, radii, conics, _num_tiles_hit, cum_tiles_hit = project(
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
        opacities=opacities,
    )
    return rasterize(
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
