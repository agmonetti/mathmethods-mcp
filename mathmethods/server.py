"""FastMCP application and tool definitions for MathematicalMethods.

Kept inside the package so it can be imported by tests and by the thin
``server.py`` entry point at the repository root.
"""

from __future__ import annotations

import math
import os

import numpy as np

from mcp.server.fastmcp import FastMCP

from mathmethods import compiler
from mathmethods.core import integration, interpolation, ode, root_finding

# --------------------------------------------------------------------------
# Resource / input caps (denial-of-service mitigation, see README security)
# --------------------------------------------------------------------------
MAX_ITERATIONS = 500
MAX_SUBINTERVALS = 2000
MAX_ODE_STEPS = 2000
MAX_INTERPOLATION_POINTS = 30
MAX_PRECISION = 15
MAX_PLOT_POINTS = 400


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _clean_float(value: float, name: str, *, positive: bool = False) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _clean_int(value: int, name: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not (minimum <= value <= maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _json_safe(value, _depth: int = 0) -> object:
    """Recursively convert a result into plain JSON-serializable data.

    NumPy scalars/arrays become Python values, NaN/Inf become ``None`` and
    anything else unsupported falls back to its string representation.
    """
    if _depth > 64:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), _depth + 1)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _downsample(values, max_points: int = MAX_PLOT_POINTS) -> object:
    if not isinstance(values, list) or len(values) <= max_points:
        return values
    step = len(values) / max_points
    return [values[int(i * step)] for i in range(max_points)]


def _finalize(result, *, downsample_keys: tuple[str, ...] = ()) -> dict:
    clean = _json_safe(result)
    for key in downsample_keys:
        if isinstance(clean, dict) and key in clean:
            clean[key] = _downsample(clean[key])
    return clean


# --------------------------------------------------------------------------
# MCP application
# --------------------------------------------------------------------------
_HTTP_HOST = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
_HTTP_PORT = int(os.environ.get("MCP_HTTP_PORT", "8000"))

mcp = FastMCP(
    "MathematicalMethods",
    host=_HTTP_HOST,
    port=_HTTP_PORT,
    instructions=(
        "Numerical-methods server: root finding (bisection), numerical "
        "integration (Simpson 1/3), ordinary differential equations "
        "(Runge-Kutta 4) and Lagrange interpolation. Math expressions use "
        "Python/SymPy syntax: x**2, sin(x), exp(x), sqrt(x), log(x). "
        "Also accepted: e^x, sen(x) and the caret (^) for powers."
    ),
)


@mcp.tool()
def root_bisection(
    func_str: str,
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    precision: int = 8,
) -> dict:
    """Find a root of f(x) = 0 in [a, b] using the bisection method.

    Requires f(a) and f(b) to have opposite signs (Bolzano's theorem).

    Args:
        func_str: Math expression in x, e.g. "x**2 - 4".
        a: Left endpoint of the interval.
        b: Right endpoint of the interval (must be > a).
        tol: Convergence tolerance on the residual/error.
        max_iter: Maximum number of iterations.
        precision: Rounding digits for the reported root.

    Returns:
        Dict with the root, convergence flag, iteration table and errors.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    tol = _clean_float(tol, "tol", positive=True)
    max_iter = _clean_int(max_iter, "max_iter", minimum=1, maximum=MAX_ITERATIONS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(b > a, "b must be greater than a.")

    normalized = compiler.validate(func_str, variables=("x",))
    func = root_finding.RootFindingService.compilar_funcion(normalized)
    result = root_finding.RootFindingService.biseccion(
        func, a, b, tol=tol, max_iter=max_iter, precision=precision
    )
    return _finalize(result)


@mcp.tool()
def integral_simpson13(
    func_str: str,
    a: float,
    b: float,
    n: int,
    epsilon: float | None = None,
    precision: int = 8,
) -> dict:
    """Approximate the definite integral of f(x) on [a, b] with composite Simpson 1/3.

    Args:
        func_str: Math expression in x, e.g. "sin(x)/x".
        a: Lower integration limit.
        b: Upper integration limit (must be > a).
        n: Number of subintervals (must be even). Error is O(h^4).
        epsilon: Point in [a, b] used to report the truncation error bound.
        precision: Rounding digits.

    Returns:
        Dict with the approximated integral, error estimates and table.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    n = _clean_int(n, "n", minimum=2, maximum=MAX_SUBINTERVALS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(b > a, "b must be greater than a.")
    _require(n % 2 == 0, "n must be even for Simpson 1/3.")
    if epsilon is not None:
        epsilon = _clean_float(epsilon, "epsilon")

    normalized = compiler.validate(func_str, variables=("x",))
    func = integration.IntegrationService.compilar_funcion(normalized)
    result = integration.IntegrationService.simpson_13_compuesto(
        func, a, b, n=n, precision=precision, epsilon=epsilon
    )
    return _finalize(result)


@mcp.tool()
def ode_rk4(
    ecuacion_str: str,
    x0: float,
    y0: float,
    xf: float,
    h: float,
    tol: float | None = None,
    precision: int = 8,
) -> dict:
    """Solve an initial value problem y'(x) = f(x, y) with the Runge-Kutta 4 method.

    Args:
        ecuacion_str: Right-hand side f(x, y), e.g. "y" or "-2*x*y".
        x0: Initial x.
        y0: Initial y(x0).
        xf: Final x (must be > x0).
        h: Step size (must be positive). The endpoint xf is always reached.
        tol: Optional tolerance to flag whether the final error meets it.
        precision: Rounding digits.

    Returns:
        Dict with the numerical solution table, exact solution (when SymPy can
        solve it) and per-step errors.
    """
    x0 = _clean_float(x0, "x0")
    y0 = _clean_float(y0, "y0")
    xf = _clean_float(xf, "xf")
    h = _clean_float(h, "h", positive=True)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(xf > x0, "xf must be greater than x0.")
    if tol is not None:
        tol = _clean_float(tol, "tol", positive=True)

    n_steps = int(round((xf - x0) / h))
    _require(1 <= n_steps <= MAX_ODE_STEPS, "step count must be between 1 and 2000.")

    normalized = compiler.validate(ecuacion_str, variables=("x", "y"))
    result = ode.ODEService.ejecutar_metodo(
        "rk4",
        normalized,
        x0,
        y0,
        xf,
        h,
        precision=precision,
        tol=tol,
    )
    return _finalize(
        result,
        downsample_keys=("x_plot", "y_plot", "y_exacta_plot"),
    )


@mcp.tool()
def interpolation_lagrange(
    puntos_x: list[float],
    x_eval: float | None = None,
    func_str: str | None = None,
    puntos_y: list[float] | None = None,
    precision: int = 8,
) -> dict:
    """Build the Lagrange interpolating polynomial through a set of points.

    Provide either the point values ``puntos_y`` or the generating function
    ``func_str`` evaluated at ``puntos_x``.

    Args:
        puntos_x: Distinct abscissas of the interpolation points.
        x_eval: Optional x where the polynomial is evaluated.
        func_str: Optional generating function f(x) for the y values.
        puntos_y: Optional y values matching ``puntos_x`` one-to-one.
        precision: Rounding digits.

    Returns:
        Dict with the polynomial, its degree, the points table and error
        bounds when ``func_str`` is provided.
    """
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(
        isinstance(puntos_x, list) and len(puntos_x) >= 2,
        "puntos_x must be a list with at least 2 points.",
    )
    n_points = len(puntos_x)
    _require(
        n_points <= MAX_INTERPOLATION_POINTS,
        f"Too many points (max {MAX_INTERPOLATION_POINTS}); Lagrange expansion grows combinatorially.",
    )

    cleaned_x = [_clean_float(xi, f"puntos_x[{i}]") for i, xi in enumerate(puntos_x)]
    _require(len(set(cleaned_x)) == n_points, "puntos_x values must be distinct.")
    puntos_x = cleaned_x

    if x_eval is not None:
        x_eval = _clean_float(x_eval, "x_eval")

    normalized_func_str = None
    if func_str is not None:
        normalized_func_str = compiler.validate(func_str, variables=("x",))
    elif puntos_y is None:
        raise ValueError("Provide either func_str or puntos_y.")

    if puntos_y is not None:
        _require(
            isinstance(puntos_y, list) and len(puntos_y) == n_points,
            "puntos_y must be a list with the same length as puntos_x.",
        )
        puntos_y = [_clean_float(yi, f"puntos_y[{i}]") for i, yi in enumerate(puntos_y)]

    result = interpolation.InterpolationService.lagrange(
        puntos_x,
        x_eval=x_eval,
        func_str=normalized_func_str,
        puntos_y=puntos_y,
        precision=precision,
    )
    return _finalize(result)


def run() -> None:
    """Run the MCP server with the transport from ``MCP_TRANSPORT`` (default stdio)."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport not in ("stdio", "streamable-http", "sse"):
        transport = "stdio"
    mcp.run(transport=transport)


if __name__ == "__main__":
    run()
