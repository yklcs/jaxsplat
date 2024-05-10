from jax.interpreters import mlir, xla
from jax.lib import xla_client
from jax import core

import functools

from jaxsplat import _jaxsplat
from jaxsplat._render import lowering, abstract


# register GPU XLA custom calls
for name, value in _jaxsplat.registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")


# forward
_render_fwd_p = core.Primitive("render_fwd")
_render_fwd_p.multiple_results = True
_render_fwd_p.def_impl(functools.partial(xla.apply_primitive, _render_fwd_p))
_render_fwd_p.def_abstract_eval(abstract._render_fwd_abs)

mlir.register_lowering(
    prim=_render_fwd_p,
    rule=lowering._render_fwd_rule,
    platform="gpu",
)

# backward
_render_bwd_p = core.Primitive("render_bwd")
_render_bwd_p.multiple_results = True
_render_bwd_p.def_impl(functools.partial(xla.apply_primitive, _render_bwd_p))
_render_bwd_p.def_abstract_eval(abstract._render_bwd_abs)

mlir.register_lowering(
    prim=_render_bwd_p,
    rule=lowering._render_bwd_rule,
    platform="gpu",
)
