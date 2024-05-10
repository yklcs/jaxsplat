from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from jaxsplat import _jaxsplat
from jaxsplat._render.abstract import (
    RenderFwdTypes,
    RenderBwdTypes,
)


def _render_fwd_rule(
    ctx: mlir.LoweringRuleContext,
    # input
    mean3ds: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    viewmat: ir.Value,
    colors: ir.Value,
    opacities: ir.Value,
    background: ir.Value,
    # desc
    num_points: int,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[int],
    block_width: int,
    clip_thresh: float,
):
    opaque = _jaxsplat.make_render_fwd_descriptor(
        num_points=num_points,
        glob_scale=glob_scale,
        f=f,
        c=c,
        img_shape=img_shape,
        block_width=block_width,
        clip_thresh=clip_thresh,
    )

    t = RenderFwdTypes(num_points)

    return custom_call(
        "render_fwd",
        operands=[
            mean3ds,
            scales,
            quats,
            viewmat,
            colors,
            opacities,
            background,
        ],
        operand_layouts=[
            t.in_mean3ds.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_viewmat.layout(),
            t.in_colors.layout(),
            t.in_opacities.layout(),
            t.in_background.layout(),
        ],
        result_types=[
            t.out_cov3ds.ir_tensor_type(),
            t.out_xys.ir_tensor_type(),
            t.out_depths.ir_tensor_type(),
            t.out_radii.ir_tensor_type(),
            t.out_conics.ir_tensor_type(),
            t.out_compensation.ir_tensor_type(),
            t.out_num_tiles_hit.ir_tensor_type(),
            t.out_gaussian_ids_sorted.ir_tensor_type(),
            t.out_tile_bins.ir_tensor_type(),
            t.out_out_img.ir_tensor_type(),
            t.out_final_Ts.ir_tensor_type(),
            t.out_final_idx.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_cov3ds.layout(),
            t.out_xys.layout(),
            t.out_depths.layout(),
            t.out_radii.layout(),
            t.out_conics.layout(),
            t.out_compensation.layout(),
            t.out_num_tiles_hit.layout(),
            t.out_gaussian_ids_sorted.layout(),
            t.out_tile_bins.layout(),
            t.out_out_img.layout(),
            t.out_final_Ts.layout(),
            t.out_final_idx.layout(),
        ],
        backend_config=opaque,
    ).results


def _render_bwd_rule(
    ctx: mlir.LoweringRuleContext,
    mean3ds: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    viewmat: ir.Value,
    colors: ir.Value,
    opacities: ir.Value,
    background: ir.Value,
    num_tiles_hit: ir.Value,
    gaussian_ids_sorted: ir.Value,
    tile_bins: ir.Value,
    final_Ts: ir.Value,
    final_idx: ir.Value,
    v_out_img: ir.Value,
    v_out_img_alpha: ir.Value,
    v_compensation: ir.Value,
    num_points: int,
    glob_scale: float,
    f: tuple[float, float],
    c: tuple[float, float],
    img_shape: tuple[float, float],
):
    opaque = _jaxsplat.make_render_bwd_descriptor(
        num_points=num_points, glob_scale=glob_scale, f=f, c=c, img_shape=img_shape
    )

    t = RenderBwdTypes(num_points)

    return custom_call(
        "project_gaussians_bwd",
        operands=[
            mean3ds,
            scales,
            quats,
            viewmat,
            colors,
            opacities,
            background,
            num_tiles_hit,
            gaussian_ids_sorted,
            tile_bins,
            final_Ts,
            final_idx,
            v_out_img,
            v_out_img_alpha,
            v_compensation,
        ],
        operand_layouts=[
            t.in_mean3ds.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_viewmat.layout(),
            t.in_colors.layout(),
            t.in_opacities.layout(),
            t.in_background.layout(),
            t.in_num_tiles_hit.layout(),
            t.in_gaussian_ids_sorted.layout(),
            t.in_tile_bins.layout(),
            t.in_final_Ts.layout(),
            t.in_final_idx.layout(),
            t.in_v_out_img.layout(),
            t.in_v_out_img_alpha.layout(),
            t.in_v_compensation.layout(),
        ],
        result_types=[
            t.out_v_mean3d.ir_tensor_type(),
            t.out_v_scale.ir_tensor_type(),
            t.out_v_quat.ir_tensor_type(),
            t.out_color.ir_tensor_type(),
            t.out_opacity.ir_tensor_type(),
            t.out_v_cov2d.ir_tensor_type(),
            t.out_v_xy.ir_tensor_type(),
            t.out_v_xy_abs.ir_tensor_type(),
            t.out_v_depth.ir_tensor_type(),
            t.out_v_conic.ir_tensor_type(),
            t.out_v_cov3d.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_v_mean3d.layout(),
            t.out_v_scale.layout(),
            t.out_v_quat.layout(),
            t.out_color.layout(),
            t.out_opacity.layout(),
            t.out_v_cov2d.layout(),
            t.out_v_xy.layout(),
            t.out_v_xy_abs.layout(),
            t.out_v_depth.layout(),
            t.out_v_conic.layout(),
            t.out_v_cov3d.layout(),
        ],
        backend_config=opaque,
    ).results
