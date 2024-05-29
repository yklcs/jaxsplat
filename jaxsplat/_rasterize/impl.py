from jax.interpreters import mlir, xla
from jax.lib import xla_client
from jax import core

import functools

import _jaxsplat
from jaxsplat._rasterize import lowering, abstract


# register GPU XLA custom calls
for name, value in _jaxsplat.registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")


# forward
_rasterize_fwd_p = core.Primitive("rasterize_fwd")
_rasterize_fwd_p.multiple_results = True
_rasterize_fwd_p.def_impl(functools.partial(xla.apply_primitive, _rasterize_fwd_p))
_rasterize_fwd_p.def_abstract_eval(abstract._rasterize_fwd_abs)

mlir.register_lowering(
    prim=_rasterize_fwd_p,
    rule=lowering._rasterize_fwd_rule,
    platform="gpu",
)

# backward
_rasterize_bwd_p = core.Primitive("rasterize_bwd")
_rasterize_bwd_p.multiple_results = True
_rasterize_bwd_p.def_impl(functools.partial(xla.apply_primitive, _rasterize_bwd_p))
_rasterize_bwd_p.def_abstract_eval(abstract._rasterize_bwd_abs)

mlir.register_lowering(
    prim=_rasterize_bwd_p,
    rule=lowering._rasterize_bwd_rule,
    platform="gpu",
)
