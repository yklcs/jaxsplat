from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from jaxsplat import _jaxsplat
from jaxsplat._rasterization.abstract import RasterizeGaussiansFwdTypes


def _rasterize_gaussians_fwd_lowering(
    ctx: mlir.LoweringRuleContext,
    # input
    xys: ir.Value,
    depths: ir.Value,
    radii: ir.Value,
    conics: ir.Value,
    num_tiles_hit: ir.Value,
    colors: ir.Value,
    opacity: ir.Value,
    # desc
    grid_dim: tuple[int, int, int],
    block_dim: tuple[int, int, int],
    img_shape: tuple[int, int, int],
):
    opaque = _jaxsplat.make_rasterize_gaussians_fwd_descriptor(
        grid_dim, block_dim, img_shape
    )

    t = RasterizeGaussiansFwdTypes(xys.shape[0])

    return custom_call(
        "rasterize_gaussians_fwd",
        operands=[xys, depths, radii, conics, num_tiles_hit, colors, opacity],
        operand_layouts=[
            t.in_xys.layout(),
            t.in_depths.layout(),
            t.in_radii.layout(),
            t.in_conics.layout(),
            t.in_num_tiles_hit.layout(),
            t.in_colors.layout(),
            t.in_opacity.layout(),
        ],
        result_types=[
            t.out_img.ir_tensor_type(),
            t.out_final_Ts.ir_tensor_type(),
            t.out_final_idx.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_img.layout(),
            t.out_final_Ts.layout(),
            t.out_final_idx.layout(),
        ],
        backend_config=opaque,
    ).results
