"""Vendored numerical-methods core.

These modules are vendored from ``modeladoYsimulacion-web/backend/app/methods``
so this MCP server is fully self-contained and publishable. Keep them in sync
with the upstream repo when that code changes.
"""

from . import (
    dynamic_1d,
    dynamic_2d_conservative,
    dynamic_2d_lanchester,
    dynamic_2d_linear,
    dynamic_2d_non_homogeneous,
    dynamic_2d_nonlinear,
    integration,
    interpolation,
    monte_carlo,
    ode,
    root_finding,
)

__all__ = [
    "dynamic_1d",
    "dynamic_2d_conservative",
    "dynamic_2d_lanchester",
    "dynamic_2d_linear",
    "dynamic_2d_non_homogeneous",
    "dynamic_2d_nonlinear",
    "integration",
    "interpolation",
    "monte_carlo",
    "ode",
    "root_finding",
]
