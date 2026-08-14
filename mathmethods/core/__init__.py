"""Vendored numerical-methods core.

These modules are vendored from ``modeladoYsimulacion-web/backend/app/methods``
so this MCP server is fully self-contained and publishable. Keep them in sync
with the upstream repo when that code changes.
"""

from . import integration, interpolation, ode, root_finding

__all__ = [
    "integration",
    "interpolation",
    "ode",
    "root_finding",
]
