import jax
import jax.numpy as jnp

from typing import TypedDict

from jaxsplat._rasterize import impl


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
    block_width: int,
) -> jax.Array:
    (img, _img_alpha) = _rasterize(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        img_shape=img_shape,
        block_width=block_width,
    )
    return img


@jax.custom_vjp
def _rasterize(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    #
    img_shape: tuple[int, int],
    block_width: int,
):
    primals, _ = _rasterize_fwd(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        img_shape=img_shape,
        block_width=block_width,
    )

    return primals


class RasterizeResiduals(TypedDict):
    colors: jax.Array
    opacities: jax.Array
    background: jax.Array
    xys: jax.Array
    depths: jax.Array
    radii: jax.Array
    conics: jax.Array
    cum_tiles_hit: jax.Array
    final_Ts: jax.Array
    final_idx: jax.Array

    num_points: int
    img_shape: tuple[int, int]
    block_width: int


def _rasterize_fwd(
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
    #
    img_shape: tuple[int, int],
    block_width: int,
):
    num_points = colors.shape[0]
    # num_intersects = cum_tiles_hit[-1]

    # if num_intersects == 0:
    #     final_Ts = jnp.zeros((*img_shape, 1), dtype=jnp.float32)
    #     final_idx = jnp.zeros((*img_shape, 1), dtype=jnp.int32)
    #     img = jnp.ones((*img_shape, 3), dtype=jnp.float32)
    # else:
    (final_Ts, final_idx, img) = impl._rasterize_fwd_p.bind(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        num_points=num_points,
        img_shape=img_shape,
        block_width=block_width,
    )

    # print("rasterize_fwd")

    # print("in")
    # print(f"  colors {colors.min():.03f} {colors.max():.03f}")
    # print(f"  opacities {opacities.min():.03f} {opacities.max():.03f}")
    # print(f"  background {background.min():.03f} {background.max():.03f}")
    # print(f"  xys {xys.min():.03f} {xys.max():.03f}")
    # print(f"  depths {depths.min():.03f} {depths.max():.03f}")
    # print(f"  radii {radii.min():.03f} {radii.max():.03f}")
    # print(f"  conics {conics.min():.03f} {conics.max():.03f}")
    # print(f"  cum_tiles_hit {cum_tiles_hit.min():.03f} {cum_tiles_hit.max():.03f}")

    # print("out")
    # print(
    # f"  gaussian_ids_sorted {gaussian_ids_sorted.min():.03f} {gaussian_ids_sorted.max():.03f}"
    # )
    # print(f"  tile_bins {tile_bins.min():.03f} {tile_bins.max():.03f}")
    # print(f"  final_Ts {final_Ts.min():.03f} {final_Ts.max():.03f}")
    # print(f"  final_idx {final_idx.min():.03f} {final_idx.max():.03f}")
    # print(f"  img {img.min():.03f} {img.max():.03f}")

    img_alpha = 1 - final_Ts
    primals = (img, img_alpha)

    residuals: RasterizeResiduals = {
        "colors": colors,
        "opacities": opacities,
        "background": background,
        "xys": xys,
        "depths": depths,
        "radii": radii,
        "conics": conics,
        "cum_tiles_hit": cum_tiles_hit,
        "final_Ts": final_Ts,
        "final_idx": final_idx,
        #
        "num_points": num_points,
        "img_shape": img_shape,
        "block_width": block_width,
    }

    return primals, residuals


def _rasterize_bwd(
    residuals: RasterizeResiduals,
    cotangents,
):
    (v_img, v_img_alpha) = cotangents

    (
        v_colors,
        v_opacity,
        v_xy,
        _v_xy_abs,
        v_conic,
    ) = impl._rasterize_bwd_p.bind(
        residuals["colors"],
        residuals["opacities"],
        residuals["background"],
        residuals["xys"],
        residuals["depths"],
        residuals["radii"],
        residuals["conics"],
        residuals["cum_tiles_hit"],
        residuals["final_Ts"],
        residuals["final_idx"],
        v_img,
        v_img_alpha,
        #
        num_points=residuals["num_points"],
        img_shape=residuals["img_shape"],
        block_width=residuals["block_width"],
    )

    # print("rasterize_bwd")

    # print("in")
    # print(f"  v_img {v_img.min():.03f} {v_img.max():.03f}")
    # print(f"  v_img_alpha {v_img_alpha.min():.03f} {v_img_alpha.max():.03f}")

    # print("out")
    # print(f"  v_colors {v_colors.min():.03f} {v_colors.max():.03f}")
    # print(f"  v_opacity {v_opacity.min():.03f} {v_opacity.max():.03f}")
    # print(f"  v_xy {v_xy.min():.03f} {v_xy.max():.03f}")
    # print(f"  v_xy_abs {_v_xy_abs.min():.03f} {_v_xy_abs.max():.03f}")
    # print(f"  v_conic {v_conic.min():.03f} {v_conic.max():.03f}")

    return (v_colors, v_opacity, None, v_xy, None, None, v_conic, None, None, None)


_rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)

__all__ = ["rasterize"]
