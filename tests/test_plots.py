"""Tests for the optional visual output (``plot=True``) and the plot registry.

Rendering tests require the ``viz`` extra (matplotlib) and are skipped when it
is not installed, so the default CI job can still run the registry and
error-path tests.
"""

import asyncio
import base64
import importlib.util
import sys

import pytest

from mathmethods import plots
from mathmethods.server import (
    dynamic_1d_bifurcation,
    dynamic_1d_solve,
    dynamic_2d_linear_solve,
    finite_differences,
    integral_comparar,
    integral_simpson13,
    interpolation_lagrange,
    mc_convergencia_1d,
    mc_estadistico_1d,
    mc_hit_or_miss_1d,
    mc_valor_promedio_1d,
    mc_valor_promedio_2d,
    mc_valor_promedio_3d,
    mcp,
    ode_rk4,
    root_bisection,
    root_comparar,
    root_newton_raphson,
    root_punto_fijo,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

requires_viz = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="requires the 'viz' extra (matplotlib)",
)


def _image_bytes(result):
    """Extract the raw PNG bytes from a ``[dict, Image]`` tool result."""
    assert isinstance(result, list), "plot=True must return a list"
    assert isinstance(result[0], dict), "the first element must remain the result dict"
    image = result[1]
    assert image.data is not None
    return image.data


# --------------------------------------------------------------------------
# Registry wiring (runs with or without matplotlib)
# --------------------------------------------------------------------------
def test_registry_covers_every_tool():
    tool_names = set(mcp._tool_manager._tools.keys())
    assert len(tool_names) == 29
    # Every tool can be plotted and there are no orphan plotters.
    assert set(plots.PLOTTERS) == tool_names


def test_plot_false_keeps_dict_contract():
    result = root_bisection(func_str="x**2 - 4", a=0, b=5)
    assert isinstance(result, dict)


def test_missing_matplotlib_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_agg", None)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    with pytest.raises(RuntimeError, match="viz"):
        plots.render("ode_rk4", {"x_plot": [0.0, 1.0], "y_plot": [1.0, 2.0]}, {})


# --------------------------------------------------------------------------
# Rendering: one representative call per plotter family
# --------------------------------------------------------------------------
PLOT_CALLS = [
    ("ode_rk4", lambda: ode_rk4("y", 0, 1, 1, 0.1, plot=True)),
    ("dynamic_1d_solve", lambda: dynamic_1d_solve(model="verhulst", params={"mu": 1.0, "K": 2.0}, plot=True)),
    (
        "dynamic_1d_bifurcation",
        lambda: dynamic_1d_bifurcation(
            model="verhulst", params={"K": 2.0}, bif_param="mu", bif_min=0, bif_max=3, bif_steps=8, plot=True
        ),
    ),
    ("dynamic_2d_linear_solve", lambda: dynamic_2d_linear_solve(cantidad_trayectorias=4, plot=True)),
    ("mc_convergencia_1d", lambda: mc_convergencia_1d("x", 0, 2, N=500, seed=1, plot=True)),
    ("mc_hit_or_miss_1d", lambda: mc_hit_or_miss_1d("x", 0, 2, N=500, seed=1, plot=True)),
    ("mc_valor_promedio_1d", lambda: mc_valor_promedio_1d("x**2", 0, 2, N=500, seed=1, plot=True)),
    ("mc_valor_promedio_2d", lambda: mc_valor_promedio_2d("x + y", 0, 1, 0, 1, N=500, seed=1, plot=True)),
    (
        "mc_valor_promedio_3d",
        lambda: mc_valor_promedio_3d("x + y + z", 0, 1, 0, 1, 0, 1, N=500, seed=1, plot=True),
    ),
    ("mc_estadistico_1d", lambda: mc_estadistico_1d("x**2", 0, 2, N=200, M=20, seed=1, plot=True)),
    ("integral_simpson13", lambda: integral_simpson13("sin(x)", 0, 3.141592653589793, 10, plot=True)),
    ("integral_comparar", lambda: integral_comparar("x**2", 0, 1, 6, plot=True)),
    ("root_bisection", lambda: root_bisection("x**2 - 4", 0, 5, plot=True)),
    ("root_newton_raphson", lambda: root_newton_raphson("x**2 - 4", 5.0, plot=True)),
    ("root_punto_fijo", lambda: root_punto_fijo("sqrt(x + 2)", 1.0, plot=True)),
    ("root_comparar", lambda: root_comparar("x**2 - 4", "sqrt(x + 2)", 0, 5, 1.0, plot=True)),
    (
        "interpolation_lagrange",
        lambda: interpolation_lagrange([0.0, 1.0, 2.0], func_str="x**2", x_eval=1.5, plot=True),
    ),
    ("finite_differences", lambda: finite_differences("x**2", 3.0, h=0.01, plot=True)),
]


@requires_viz
@pytest.mark.parametrize("name,call", PLOT_CALLS, ids=[c[0] for c in PLOT_CALLS])
def test_plot_true_returns_png(name, call):
    result = call()
    assert isinstance(result, list) and len(result) == 2
    assert isinstance(result[0], dict)
    assert _image_bytes(result).startswith(PNG_SIGNATURE)


@requires_viz
def test_render_is_deterministic():
    result = ode_rk4("y", 0, 1, 1, 0.1)
    first = plots.render("ode_rk4", result, {"metodo": "RK4"})
    second = plots.render("ode_rk4", result, {"metodo": "RK4"})
    assert first == second
    assert first.startswith(PNG_SIGNATURE)


# Scalar-valued SymPy functions (gamma, erf, factorial, ...) map to non
# vectorized callables; the plotter must fall back to per-point evaluation.
SCALAR_PLOT_CALLS = [
    ("erf", lambda: root_bisection("erf(x) - 0.5", 0, 2, plot=True)),
    ("factorial", lambda: root_newton_raphson("factorial(x) - 6", 2.0, plot=True)),
    ("gamma", lambda: finite_differences("gamma(x)", 4.0, plot=True)),
]


@requires_viz
@pytest.mark.parametrize("name,call", SCALAR_PLOT_CALLS, ids=[c[0] for c in SCALAR_PLOT_CALLS])
def test_scalar_map_functions_render(name, call):
    result = call()
    assert isinstance(result, list) and len(result) == 2
    assert _image_bytes(result).startswith(PNG_SIGNATURE)


def test_trim_large_arrays_downsamples_with_none_entries():
    from mathmethods.server import _trim_large_arrays

    trimmed = _trim_large_arrays([1.0] * 1999 + [None])
    assert len(trimmed) <= 400


# --------------------------------------------------------------------------
# End-to-end over the MCP protocol: JSON text + image content
# --------------------------------------------------------------------------
@requires_viz
def test_plot_true_over_protocol_emits_image_content():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def main():
        async with create_connected_server_and_client_session(mcp._mcp_server) as session:
            res = await session.call_tool(
                "ode_rk4",
                {"ecuacion_str": "y", "x0": 0, "y0": 1, "xf": 1, "h": 0.2, "plot": True},
            )
            assert res.isError is False
            types = [c.type for c in res.content]
            assert "text" in types and "image" in types
            image = next(c for c in res.content if c.type == "image")
            assert image.mimeType == "image/png"
            assert base64.b64decode(image.data).startswith(PNG_SIGNATURE)

    asyncio.run(main())


@requires_viz
def test_plot_false_over_protocol_emits_text_only():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def main():
        async with create_connected_server_and_client_session(mcp._mcp_server) as session:
            res = await session.call_tool(
                "ode_rk4",
                {"ecuacion_str": "y", "x0": 0, "y0": 1, "xf": 1, "h": 0.2},
            )
            assert res.isError is False
            assert [c.type for c in res.content] == ["text"]

    asyncio.run(main())
