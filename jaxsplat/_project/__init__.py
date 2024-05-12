import jax

from typing import TypedDict

from jaxsplat._project import impl


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
    (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit, _compensation) = (
        _project(
            mean3ds,
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
    )
    return (xys, depths, radii, conics, num_tiles_hit, cum_tiles_hit)


@jax.custom_vjp
def _project(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    #
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    primals, _ = _project_fwd(
        mean3ds,
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

    num_points: int
    img_shape: tuple[int, int]
    f: tuple[float, float]
    c: tuple[float, float]
    glob_scale: float
    clip_thresh: float
    block_width: int


def _project_fwd(
    mean3ds: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    viewmat: jax.Array,
    #
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    num_points = mean3ds.shape[0]

    (cov3ds, xys, depths, radii, conics, compensation, num_tiles_hit, cum_tiles_hit) = (
        impl._project_fwd_p.bind(
            mean3ds,
            scales,
            quats,
            viewmat,
            num_points=num_points,
            img_shape=img_shape,
            f=f,
            c=c,
            glob_scale=glob_scale,
            clip_thresh=clip_thresh,
            block_width=block_width,
        )
    )

    # print("project_fwd")

    # print("in")
    # print(f"  mean3ds {mean3ds.min():.03f} {mean3ds.max():.03f}")
    # print(f"  scales {scales.min():.03f} {scales.max():.03f}")
    # print(f"  quats {quats.min():.03f} {quats.max():.03f}")
    # print(f"  viewmat {viewmat.min():.03f} {viewmat.max():.03f}")

    # print("out")
    # print(f"  cov3ds {cov3ds.min():.03f} {cov3ds.max():.03f}")
    # print(f"  xys {xys.min():.03f} {xys.max():.03f}")
    # print(f"  depths {depths.min():.03f} {depths.max():.03f}")
    # print(f"  radii {radii.min():.03f} {radii.max():.03f}")
    # print(f"  conics {conics.min():.03f} {conics.max():.03f}")
    # print(f"  compensation {compensation.min():.03f} {compensation.max():.03f}")
    # print(f"  num_tiles_hit {num_tiles_hit.min():.03f} {num_tiles_hit.max():.03f}")
    # print(f"  cum_tiles_hit {cum_tiles_hit.min():.03f} {cum_tiles_hit.max():.03f}")

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
        #
        "num_points": num_points,
        "img_shape": img_shape,
        "f": f,
        "c": c,
        "glob_scale": glob_scale,
        "clip_thresh": clip_thresh,
        "block_width": block_width,
    }

    return primals, residuals


def _project_bwd(
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

    num_points = residuals["mean3ds"].shape[0]

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
        num_points=num_points,
        img_shape=residuals["img_shape"],
        f=residuals["f"],
        c=residuals["c"],
        glob_scale=residuals["glob_scale"],
        clip_thresh=residuals["clip_thresh"],
        block_width=residuals["block_width"],
    )

    # print("project_bwd")

    # print("in")
    # print(f"  v_xy {v_xy.min():.03f} {v_xy.max():.03f}")
    # print(f"  v_depth {v_depth.min():.03f} {v_depth.max():.03f}")
    # print(f"  v_conic {v_conic.min():.03f} {v_conic.max():.03f}")
    # print(f"  v_compensation {v_compensation.min():.03f} {v_compensation.max():.03f}")

    # print("out")
    # print(f"  v_mean3d {v_mean3d.min():.03f} {v_mean3d.max():.03f}")
    # print(f"  v_scale {v_scale.min():.03f} {v_scale.max():.03f}")
    # print(f"  v_quat {v_quat.min():.03f} {v_quat.max():.03f}")
    # print(f"  v_cov2d {v_cov2d.min():.03f} {v_cov2d.max():.03f}")
    # print(f"  v_cov3d {v_cov3d.min():.03f} {v_cov3d.max():.03f}")

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


_project.defvjp(_project_fwd, _project_bwd)

__all__ = ["project"]
