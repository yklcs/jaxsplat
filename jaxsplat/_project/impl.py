from jax.interpreters import mlir, xla
from jax.lib import xla_client
from jax import core

import functools

from jaxsplat import _jaxsplat
from jaxsplat._project import lowering, abstract


# register GPU XLA custom calls
for name, value in _jaxsplat.registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")


# forward
_project_fwd_p = core.Primitive("project_fwd")
_project_fwd_p.multiple_results = True
_project_fwd_p.def_impl(functools.partial(xla.apply_primitive, _project_fwd_p))
_project_fwd_p.def_abstract_eval(abstract._project_fwd_abs)

mlir.register_lowering(
    prim=_project_fwd_p,
    rule=lowering._project_fwd_rule,
    platform="gpu",
)

# backward
_project_bwd_p = core.Primitive("project_bwd")
_project_bwd_p.multiple_results = True
_project_bwd_p.def_impl(functools.partial(xla.apply_primitive, _project_bwd_p))
_project_bwd_p.def_abstract_eval(abstract._project_bwd_abs)

mlir.register_lowering(
    prim=_project_bwd_p,
    rule=lowering._project_bwd_rule,
    platform="gpu",
)
