from jax.interpreters import mlir, xla
from jax.lib import xla_client
from jax import core

import functools

from jaxsplat import _jaxsplat
from jaxsplat._rasterization import lowering, abstract


# register GPU XLA custom calls
for name, value in _jaxsplat.rasterization_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")


# forward
_rasterize_gaussians_fwd_p = core.Primitive("rasterize_gaussians_fwd")
_rasterize_gaussians_fwd_p.multiple_results = True
_rasterize_gaussians_fwd_p.def_impl(
    functools.partial(xla.apply_primitive, _rasterize_gaussians_fwd_p)
)
_rasterize_gaussians_fwd_p.def_abstract_eval(abstract._rasterize_gaussians_fwd_abs)

mlir.register_lowering(
    prim=_rasterize_gaussians_fwd_p,
    rule=lowering._rasterize_gaussians_fwd_lowering,
    platform="gpu",
)
