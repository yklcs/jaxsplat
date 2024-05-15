import jax

from typing import TypedDict
from dataclasses import dataclass
from functools import partial

from jaxsplat._project import impl


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class ProjectDescriptor:
    num_points: int
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float
    block_width: int

    def tree_flatten(self):
        children = ()
        aux = (
            self.num_points,
            self.img_shape,
            self.f,
            self.c,
            self.glob_scale,
            self.clip_thresh,
            self.block_width,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            num_points,
            img_shape,
            f,
            c,
            glob_scale,
            clip_thresh,
            block_width,
        ) = aux
        return cls(
            num_points=num_points,
            img_shape=img_shape,
            f=f,
            c=c,
            glob_scale=glob_scale,
            clip_thresh=clip_thresh,
            block_width=block_width,
        )


def project(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    *,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    desc = ProjectDescriptor(
        num_points=mean3ds.shape[0],
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )
    (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit, _compensation) = (
        _project(
            desc,
            mean3ds,
            scales,
            quats,
            viewmat,
        )
    )
    return (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _project(
    desc: ProjectDescriptor,
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
):
    primals, _ = _project_fwd(desc, mean3ds, scales, quats, viewmat)

    return primals


class ProjectResiduals(TypedDict):
    mean3ds: jax.Array
    scales: jax.Array
    quats: jax.Array
    viewmat: jax.Array
    cov3ds: jax.Array
    xys: jax.Array
    radii: jax.Array
    conics: jax.Array
    compensation: jax.Array


def _project_fwd(
    desc: ProjectDescriptor,
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
):
    (cov3ds, xys, depths, radii, conics, compensation, num_tiles_hit, cum_tiles_hit) = (
        impl._project_fwd_p.bind(
            mean3ds,
            scales,
            quats,
            viewmat,
            num_points=desc.num_points,
            img_shape=desc.img_shape,
            f=desc.f,
            c=desc.c,
            glob_scale=desc.glob_scale,
            clip_thresh=desc.clip_thresh,
            block_width=desc.block_width,
        )
    )

    primals = (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit, compensation)

    residuals: ProjectResiduals = {
        "mean3ds": mean3ds,
        "scales": scales,
        "quats": quats,
        "viewmat": viewmat,
        "cov3ds": cov3ds,
        "xys": xys,
        "radii": radii,
        "conics": conics,
        "compensation": compensation,
    }

    return primals, residuals


def _project_bwd(
    desc: ProjectDescriptor,
    residuals: ProjectResiduals,
    cotangents,
):
    (
        v_xy,
        v_depth,
        _v_radii,
        v_conic,
        _v_num_tiles_hit,
        _v_cum_tiles_hit,
        v_compensation,
    ) = cotangents

    (
        v_mean3d,
        v_scale,
        v_quat,
        v_cov2d,
        v_cov3d,
    ) = impl._project_bwd_p.bind(
        residuals["mean3ds"],
        residuals["scales"],
        residuals["quats"],
        residuals["viewmat"],
        residuals["cov3ds"],
        residuals["xys"],
        residuals["radii"],
        residuals["conics"],
        residuals["compensation"],
        v_compensation,
        v_xy,
        v_depth,
        v_conic,
        #
        num_points=desc.num_points,
        img_shape=desc.img_shape,
        f=desc.f,
        c=desc.c,
        glob_scale=desc.glob_scale,
        clip_thresh=desc.clip_thresh,
        block_width=desc.block_width,
    )

    return (
        v_mean3d,
        v_scale,
        v_quat,
        None,
    )


_project.defvjp(_project_fwd, _project_bwd)

__all__ = ["project"]
