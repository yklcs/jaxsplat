import jax

from typing import TypedDict
from dataclasses import dataclass
from functools import partial

from jaxsplat._rasterize import impl


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class RasterizeDescriptor:
    num_points: int
    img_shape: tuple[int, int]
    block_width: int

    def tree_flatten(self):
        children = ()
        aux = (
            self.num_points,
            self.img_shape,
            self.block_width,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            num_points,
            img_shape,
            block_width,
        ) = aux
        return cls(
            num_points=num_points,
            img_shape=img_shape,
            block_width=block_width,
        )


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
    desc = RasterizeDescriptor(
        num_points=colors.shape[0], img_shape=img_shape, block_width=block_width
    )

    (img, _img_alpha) = _rasterize(
        desc,
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
    )
    return img


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _rasterize(
    desc: RasterizeDescriptor,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
):
    primals, _ = _rasterize_fwd(
        desc,
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
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


def _rasterize_fwd(
    desc: RasterizeDescriptor,
    colors: jax.Array,
    opacities: jax.Array,
    background: jax.Array,
    xys: jax.Array,
    depths: jax.Array,
    radii: jax.Array,
    conics: jax.Array,
    cum_tiles_hit: jax.Array,
):
    (final_Ts, final_idx, img) = impl._rasterize_fwd_p.bind(
        colors,
        opacities,
        background,
        xys,
        depths,
        radii,
        conics,
        cum_tiles_hit,
        num_points=desc.num_points,
        img_shape=desc.img_shape,
        block_width=desc.block_width,
    )

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
    }

    return primals, residuals


def _rasterize_bwd(
    desc: RasterizeDescriptor,
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
        num_points=desc.num_points,
        img_shape=desc.img_shape,
        block_width=desc.block_width,
    )

    return (v_colors, v_opacity, None, v_xy, None, None, v_conic, None)


_rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)

__all__ = ["rasterize"]
