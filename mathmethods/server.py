"""FastMCP application and tool definitions for MathematicalMethods.

Kept inside the package so it can be imported by tests and by the thin
``server.py`` entry point at the repository root.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable

import numpy as np
from mcp.server.fastmcp import FastMCP

from mathmethods import compiler
from mathmethods.core import (
    differentiation,
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


def _trim_large_arrays(value) -> object:
    """Recursively downsample long numeric lists to keep responses bounded."""
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float)) for v in value):
            return _downsample(value)
        return [_trim_large_arrays(v) for v in value]
    if isinstance(value, dict):
        return {k: _trim_large_arrays(v) for k, v in value.items()}
    return value


def _finalize_deep(result) -> dict:
    return _trim_large_arrays(_json_safe(result))


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
    return _ode_solve("rk4", ecuacion_str, x0, y0, xf, h, tol, precision)


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


# --------------------------------------------------------------------------
# Additional caps for the dynamic / Monte Carlo tools
# --------------------------------------------------------------------------
MAX_MC_N = 200_000
MAX_MC_M = 500
MAX_DYN_GRID = 2000
MAX_DYN_STEPS = 5000
MAX_TRAJECTORIES = 50
MAX_BIF_STEPS = 500


# --------------------------------------------------------------------------
# Monte Carlo tools
# --------------------------------------------------------------------------
@mcp.tool()
def mc_hit_or_miss_1d(
    func_str: str,
    a: float,
    b: float,
    N: int = 10000,
    seed: int | None = None,
    precision: int = 8,
    nivel_confianza: float = 0.95,
) -> dict:
    """Estimate ∫ₐᵇ f(x) dx with the hit-or-miss Monte Carlo method.

    Handles sign-changing integrands correctly (returns the signed integral).

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        N: Number of samples.
        seed: Optional RNG seed for reproducibility.
        precision: Rounding digits.
        nivel_confianza: Confidence level in (0, 1) for the interval.

    Returns:
        Dict with the estimate, interval, sample statistics and history.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    N = _clean_int(N, "N", minimum=1, maximum=MAX_MC_N)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    nivel_confianza = _clean_float(nivel_confianza, "nivel_confianza")
    _require(0 < nivel_confianza < 1, "nivel_confianza must be in (0, 1).")
    _require(b > a, "b must be greater than a.")
    if seed is not None:
        seed = _clean_int(seed, "seed", minimum=0, maximum=2**31 - 1)

    normalized = compiler.validate(func_str, variables=("x",))
    func = monte_carlo.MonteCarloService.compilar_funcion(normalized, variables="x")
    result = monte_carlo.MonteCarloService.hit_or_miss_1d(
        func, a, b, N=N, seed=seed, precision=precision, nivel_confianza=nivel_confianza
    )
    result["historial"] = _downsample(result.get("historial") or [], 100)
    return _finalize_deep(result)


@mcp.tool()
def mc_valor_promedio_1d(
    func_str: str,
    a: float,
    b: float,
    N: int = 10000,
    seed: int | None = None,
    precision: int = 8,
    nivel_confianza: float = 0.95,
) -> dict:
    """Estimate ∫ₐᵇ f(x) dx with the mean-value Monte Carlo method.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        N: Number of samples.
        seed: Optional RNG seed for reproducibility.
        precision: Rounding digits.
        nivel_confianza: Confidence level in (0, 1).

    Returns:
        Dict with the estimate, confidence interval and sample statistics.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    N = _clean_int(N, "N", minimum=1, maximum=MAX_MC_N)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    nivel_confianza = _clean_float(nivel_confianza, "nivel_confianza")
    _require(0 < nivel_confianza < 1, "nivel_confianza must be in (0, 1).")
    _require(b > a, "b must be greater than a.")
    if seed is not None:
        seed = _clean_int(seed, "seed", minimum=0, maximum=2**31 - 1)

    normalized = compiler.validate(func_str, variables=("x",))
    func = monte_carlo.MonteCarloService.compilar_funcion(normalized, variables="x")
    result = monte_carlo.MonteCarloService.valor_promedio_1d(
        func, a, b, N=N, seed=seed, precision=precision, nivel_confianza=nivel_confianza
    )
    return _finalize_deep(result)


@mcp.tool()
def mc_valor_promedio_2d(
    func_str: str,
    x_a: float,
    x_b: float,
    y_a: float,
    y_b: float,
    N: int = 10000,
    seed: int | None = None,
    precision: int = 8,
    nivel_confianza: float = 0.95,
) -> dict:
    """Estimate the double integral of f(x, y) over [x_a,x_b]×[y_a,y_b] by mean value."""
    x_a, x_b = _clean_float(x_a, "x_a"), _clean_float(x_b, "x_b")
    y_a, y_b = _clean_float(y_a, "y_a"), _clean_float(y_b, "y_b")
    N = _clean_int(N, "N", minimum=1, maximum=MAX_MC_N)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    nivel_confianza = _clean_float(nivel_confianza, "nivel_confianza")
    _require(0 < nivel_confianza < 1, "nivel_confianza must be in (0, 1).")
    _require(x_b > x_a and y_b > y_a, "upper limits must be greater than lower limits.")
    if seed is not None:
        seed = _clean_int(seed, "seed", minimum=0, maximum=2**31 - 1)

    normalized = compiler.validate(func_str, variables=("x", "y"))
    func = monte_carlo.MonteCarloService.compilar_funcion(normalized, variables="x y")
    result = monte_carlo.MonteCarloService.valor_promedio_2d(
        func, (x_a, x_b), (y_a, y_b), N=N, seed=seed, precision=precision, nivel_confianza=nivel_confianza
    )
    return _finalize_deep(result)


@mcp.tool()
def mc_valor_promedio_3d(
    func_str: str,
    x_a: float,
    x_b: float,
    y_a: float,
    y_b: float,
    z_a: float,
    z_b: float,
    N: int = 10000,
    seed: int | None = None,
    precision: int = 8,
    nivel_confianza: float = 0.95,
) -> dict:
    """Estimate the triple integral of f(x,y,z) over the box [x_a,x_b]×[y_a,y_b]×[z_a,z_b]."""
    x_a, x_b = _clean_float(x_a, "x_a"), _clean_float(x_b, "x_b")
    y_a, y_b = _clean_float(y_a, "y_a"), _clean_float(y_b, "y_b")
    z_a, z_b = _clean_float(z_a, "z_a"), _clean_float(z_b, "z_b")
    N = _clean_int(N, "N", minimum=1, maximum=MAX_MC_N)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    nivel_confianza = _clean_float(nivel_confianza, "nivel_confianza")
    _require(0 < nivel_confianza < 1, "nivel_confianza must be in (0, 1).")
    _require(x_b > x_a and y_b > y_a and z_b > z_a, "upper limits must be greater than lower limits.")
    if seed is not None:
        seed = _clean_int(seed, "seed", minimum=0, maximum=2**31 - 1)

    normalized = compiler.validate(func_str, variables=("x", "y", "z"))
    func = monte_carlo.MonteCarloService.compilar_funcion(normalized, variables="x y z")
    result = monte_carlo.MonteCarloService.valor_promedio_3d(
        func,
        (x_a, x_b),
        (y_a, y_b),
        (z_a, z_b),
        N=N,
        seed=seed,
        precision=precision,
        nivel_confianza=nivel_confianza,
    )
    return _finalize_deep(result)


@mcp.tool()
def mc_estadistico_1d(
    func_str: str,
    a: float,
    b: float,
    N: int = 1000,
    M: int = 100,
    nivel_confianza: float = 0.95,
    seed: int | None = None,
    precision: int = 8,
) -> dict:
    """Replicated Monte Carlo experiment (M repetitions of N samples) with statistical analysis.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        N: Samples per replication.
        M: Number of replications.
        nivel_confianza: Confidence level in (0, 1).
        seed: Optional RNG seed.
        precision: Rounding digits.

    Returns:
        Dict with the distribution of estimates and confidence intervals.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    N = _clean_int(N, "N", minimum=2, maximum=MAX_MC_N // 10)
    M = _clean_int(M, "M", minimum=2, maximum=MAX_MC_M)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    nivel_confianza = _clean_float(nivel_confianza, "nivel_confianza")
    _require(0 < nivel_confianza < 1, "nivel_confianza must be in (0, 1).")
    _require(b > a, "b must be greater than a.")
    if seed is not None:
        seed = _clean_int(seed, "seed", minimum=0, maximum=2**31 - 1)

    normalized = compiler.validate(func_str, variables=("x",))
    func = monte_carlo.MonteCarloService.compilar_funcion(normalized, variables="x")
    result = monte_carlo.MonteCarloService.analisis_estadistico_1d(
        func, a, b, N=N, M=M, nivel_confianza=nivel_confianza, seed=seed, precision=precision
    )
    return _finalize_deep(result)


# --------------------------------------------------------------------------
# Dynamic 1D tools
# --------------------------------------------------------------------------
@mcp.tool()
def dynamic_1d_solve(
    func_str: str = "x",
    model: str = "custom",
    params: dict[str, float] | None = None,
    x_min: float = -1.0,
    x_max: float = 3.0,
    t_max: float = 10.0,
    n_phase: int = 400,
    n_time: int = 200,
    initial_conditions: list[float] | None = None,
    control_enabled: bool = False,
) -> dict:
    """Analyze the 1D autonomous system x' = f(x): equilibria, stability, phase portrait and time series.

    Args:
        func_str: f(x) used when model is 'custom'.
        model: 'custom' | 'malthus' (r*x) | 'verhulst' (mu*x*(1-x/K)) | 'newton' (-k*(x-Ta)).
        params: parameter values, e.g. {'r': 1.5} or {'mu': 1.0, 'K': 2.0}.
        x_min, x_max: state window.
        t_max: final time of the time series.
        n_phase: resolution of the phase analysis.
        n_time: number of time steps.
        initial_conditions: starting states, e.g. [0.1, 1.0].
        control_enabled: add a constant control term -h (verhulst/custom only).

    Returns:
        Dict with equilibria, their stability, the phase portrait data and time solutions.
    """
    params = params or {}
    x_min = _clean_float(x_min, "x_min")
    x_max = _clean_float(x_max, "x_max")
    t_max = _clean_float(t_max, "t_max", positive=True)
    n_phase = _clean_int(n_phase, "n_phase", minimum=10, maximum=MAX_DYN_GRID)
    n_time = _clean_int(n_time, "n_time", minimum=2, maximum=MAX_DYN_GRID)
    _require(x_max > x_min, "x_max must be greater than x_min.")
    if initial_conditions is None:
        initial_conditions = [0.5]

    if model == "custom":
        compiler.validate(func_str, variables=("x", *params.keys()))
    elif model not in ("malthus", "verhulst", "newton"):
        raise ValueError("model must be one of: custom, malthus, verhulst, newton.")

    payload = {
        "model": model,
        "func_str": func_str,
        "params": params,
        "control_enabled": control_enabled,
        "x_min": x_min,
        "x_max": x_max,
        "t_max": t_max,
        "n_phase": n_phase,
        "n_time": n_time,
        "initial_conditions": initial_conditions,
    }
    result = dynamic_1d.Dynamic1DService.solve(payload)
    return _finalize_deep(result)


@mcp.tool()
def dynamic_1d_equilibria(
    func_str: str = "x",
    model: str = "custom",
    params: dict[str, float] | None = None,
    x_min: float = -1.0,
    x_max: float = 3.0,
    n_phase: int = 400,
    control_enabled: bool = False,
) -> dict:
    """Find the equilibria of x' = f(x) and classify their stability."""
    params = params or {}
    x_min = _clean_float(x_min, "x_min")
    x_max = _clean_float(x_max, "x_max")
    n_phase = _clean_int(n_phase, "n_phase", minimum=10, maximum=MAX_DYN_GRID)
    _require(x_max > x_min, "x_max must be greater than x_min.")

    if model == "custom":
        compiler.validate(func_str, variables=("x", *params.keys()))
    elif model not in ("malthus", "verhulst", "newton"):
        raise ValueError("model must be one of: custom, malthus, verhulst, newton.")

    payload = {
        "model": model,
        "func_str": func_str,
        "params": params,
        "control_enabled": control_enabled,
        "x_min": x_min,
        "x_max": x_max,
        "n_phase": n_phase,
    }
    result = dynamic_1d.Dynamic1DService.solve(payload)
    return _finalize_deep(result)


@mcp.tool()
def dynamic_1d_bifurcation(
    func_str: str = "x",
    model: str = "custom",
    params: dict[str, float] | None = None,
    bif_param: str = "r",
    bif_min: float = -1.0,
    bif_max: float = 1.0,
    bif_steps: int = 60,
    x_min: float = -1.0,
    x_max: float = 3.0,
    n_phase: int = 400,
    phase_params: list[float] | None = None,
    control_enabled: bool = False,
) -> dict:
    """Bifurcation analysis of x' = f(x; bif_param): equilibria vs parameter sweep.

    Args:
        func_str: f(x) used when model is 'custom'.
        model: 'custom' | 'malthus' | 'verhulst' | 'newton'.
        params: values of the other model parameters, e.g. {'K': 2.0}.
        bif_param: name of the bifurcation parameter swept between bif_min and bif_max.
        bif_min, bif_max, bif_steps: parameter sweep range/resolution.
        x_min, x_max, n_phase: state window for finding equilibria.
        phase_params: optional explicit parameter values for phase slices.
        control_enabled: add a constant control term -h (verhulst/custom only).

    Returns:
        Dict with the equilibria-vs-parameter table, exact symbolic analysis and phase slices.
    """
    params = params or {}
    bif_min = _clean_float(bif_min, "bif_min")
    bif_max = _clean_float(bif_max, "bif_max")
    bif_steps = _clean_int(bif_steps, "bif_steps", minimum=2, maximum=MAX_BIF_STEPS)
    x_min = _clean_float(x_min, "x_min")
    x_max = _clean_float(x_max, "x_max")
    n_phase = _clean_int(n_phase, "n_phase", minimum=10, maximum=MAX_DYN_GRID)
    _require(bif_max > bif_min, "bif_max must be greater than bif_min.")
    _require(x_max > x_min, "x_max must be greater than x_min.")

    if model == "custom":
        compiler.validate(func_str, variables=("x", bif_param, *params.keys()))
    elif model not in ("malthus", "verhulst", "newton"):
        raise ValueError("model must be one of: custom, malthus, verhulst, newton.")
    _require(
        bool(bif_param.strip()) and bif_param.strip().isidentifier(),
        "bif_param must be a valid symbol name.",
    )

    payload = {
        "model": model,
        "func_str": func_str,
        "params": params,
        "control_enabled": control_enabled,
        "bif_param": bif_param,
        "bif_min": bif_min,
        "bif_max": bif_max,
        "bif_steps": bif_steps,
        "x_min": x_min,
        "x_max": x_max,
        "n_phase": n_phase,
        "phase_params": phase_params,
    }
    result = dynamic_1d.Dynamic1DService.bifurcation(payload)
    result = _finalize_deep(result)
    # The equilibria-vs-parameter table can be huge; keep it digestible for the LLM.
    bif = result.get("bifurcation")
    if isinstance(bif, dict):
        bif["equilibria"] = _downsample(bif.get("equilibria") or [], 300)
    return result


# --------------------------------------------------------------------------
# Dynamic 2D tools
# --------------------------------------------------------------------------
def _clean_dynamic2d_inputs(x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, traj):
    x0 = _clean_float(x0, "x0")
    y0 = _clean_float(y0, "y0")
    t0 = _clean_float(t0, "t0")
    t_fin = _clean_float(t_fin, "t_fin")
    h = _clean_float(h, "h", positive=True)
    x_min = _clean_float(x_min, "x_min")
    x_max = _clean_float(x_max, "x_max")
    y_min = _clean_float(y_min, "y_min")
    y_max = _clean_float(y_max, "y_max")
    traj = _clean_int(traj, "cantidad_trayectorias", minimum=1, maximum=MAX_TRAJECTORIES)
    _require(t_fin > t0, "t_fin must be greater than t0.")
    _require(x_max > x_min and y_max > y_min, "max limits must be greater than min limits.")
    steps = round((t_fin - t0) / h)
    _require(1 <= steps <= MAX_DYN_STEPS, "step count must be between 1 and 5000.")
    return x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, traj


@mcp.tool()
def dynamic_2d_linear_solve(
    a: float = 3.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 3.0,
    e: float = 0.0,
    f: float = 0.0,
    x0: float = 1.0,
    y0: float = 1.0,
    t0: float = 0.0,
    t_fin: float = 10.0,
    h: float = 0.01,
    x_min: float = -5.0,
    x_max: float = 5.0,
    y_min: float = -5.0,
    y_max: float = 5.0,
    cantidad_trayectorias: int = 16,
) -> dict:
    """Solve and classify the linear 2D system X' = A·X + B with constant matrix A.

    Args:
        a, b, c, d: entries of A = [[a, b], [c, d]].
        e, f: constant forcing vector B = [e, f].
        x0, y0, t0, t_fin, h: initial condition and integration grid.
        x_min..y_max: window for the phase portrait.
        cantidad_trayectorias: number of sample trajectories.

    Returns:
        Dict with the classification, eigenvalues, nullclines, analytic solution and trajectories.
    """
    x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, traj = _clean_dynamic2d_inputs(
        x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, cantidad_trayectorias
    )
    payload = {
        "a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
        "x0": x0, "y0": y0, "t0": t0, "t_fin": t_fin, "h": h,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "cantidad_trayectorias": traj,
    }
    result = dynamic_2d_linear.Dynamic2DLinearService.solve(payload)
    return _finalize_deep(result)


@mcp.tool()
def dynamic_2d_nonlinear_solve(
    eq_x: str = "y - x",
    eq_y: str = "x**2 - 1",
    params: dict[str, float] | None = None,
    mu: float = 0.0,
    x0: float = 0.5,
    y0: float = 0.5,
    t0: float = 0.0,
    t_fin: float = 10.0,
    h: float = 0.02,
    x_min: float = -3.0,
    x_max: float = 3.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
    cantidad_trayectorias: int = 25,
) -> dict:
    """Solve and analyze the nonlinear 2D system x' = f(x,y), y' = g(x,y).

    Args:
        eq_x, eq_y: Math expressions for f and g (variables x, y, mu and extra params).
        params: extra parameter values (e.g. {'alpha': 0.5}).
        mu: bifurcation parameter value.
        x0..y_max, cantidad_trayectorias: integration and portrait settings.

    Returns:
        Dict with equilibria, Jacobian-based classification, nullclines and trajectories.
    """
    params = params or {}
    x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, traj = _clean_dynamic2d_inputs(
        x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, cantidad_trayectorias
    )
    gate = ["x", "y", "mu", *list(params.keys())]
    compiler.validate(eq_x, variables=tuple(dict.fromkeys(gate)))
    compiler.validate(eq_y, variables=tuple(dict.fromkeys(gate)))
    payload = {
        "eq_x": eq_x, "eq_y": eq_y, "params": params, "mu": mu,
        "x0": x0, "y0": y0, "t0": t0, "t_fin": t_fin, "h": h,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "cantidad_trayectorias": traj,
    }
    result = dynamic_2d_nonlinear.Dynamic2DNonLinearService.solve(payload)
    return _finalize_deep(result)


@mcp.tool()
def dynamic_2d_conservative_solve(
    eq_x: str = "y",
    eq_y: str = "x - x**3",
    mu: float = 0.0,
    x0: float = 0.1,
    y0: float = 0.0,
    t0: float = 0.0,
    t_fin: float = 15.0,
    h: float = 0.02,
    x_min: float = -2.5,
    x_max: float = 2.5,
    y_min: float = -2.5,
    y_max: float = 2.5,
    cantidad_trayectorias: int = 25,
) -> dict:
    """Analyze a 2D system checking conservativeness (divergence-free), Hamiltonian/energy and closed orbits.

    Args:
        eq_x, eq_y: Math expressions for x' and y'.
        mu: parameter value.
        x0..y_max, cantidad_trayectorias: integration and portrait settings.

    Returns:
        Dict with the divergence check, the Hamiltonian reconstruction, the
        equilibrium classification and the trajectories.
    """
    x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, traj = _clean_dynamic2d_inputs(
        x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, cantidad_trayectorias
    )
    compiler.validate(eq_x, variables=("x", "y", "mu"))
    compiler.validate(eq_y, variables=("x", "y", "mu"))
    payload = {
        "eq_x": eq_x, "eq_y": eq_y, "mu": mu,
        "x0": x0, "y0": y0, "t0": t0, "t_fin": t_fin, "h": h,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "cantidad_trayectorias": traj,
    }
    result = dynamic_2d_conservative.Dynamic2DConservativeService.solve(payload)
    return _finalize_deep(result)


@mcp.tool()
def dynamic_2d_lanchester_solve(
    eq_x: str = "-α * y",
    eq_y: str = "-β * x",
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    mu: float = 0.0,
    delta: float = 0.0,
    x0: float = 100.0,
    y0: float = 80.0,
    t0: float = 0.0,
    t_fin: float = 5.0,
    h: float = 0.01,
) -> dict:
    """Simulate a Lanchester combat model x' = f(x,y), y' = g(x,y) with analytic time-to-annihilation.

    Args:
        eq_x, eq_y: Math expressions (classic default is x'=-αy, y'=-βx).
        alpha, beta, gamma, epsilon, mu, delta: Greek parameter values.
        x0, y0: initial force sizes.
        t0, t_fin, h: simulation grid.

    Returns:
        Dict with the analytic winner/survivors, state equation and the numerical trajectories.
    """
    alpha = _clean_float(alpha, "alpha")
    beta = _clean_float(beta, "beta")
    gamma = _clean_float(gamma, "gamma")
    epsilon = _clean_float(epsilon, "epsilon")
    mu = _clean_float(mu, "mu")
    delta = _clean_float(delta, "delta")
    x0 = _clean_float(x0, "x0")
    y0 = _clean_float(y0, "y0")
    t0 = _clean_float(t0, "t0")
    t_fin = _clean_float(t_fin, "t_fin")
    h = _clean_float(h, "h", positive=True)
    _require(t_fin > t0, "t_fin must be greater than t0.")
    steps = round((t_fin - t0) / h)
    _require(1 <= steps <= MAX_DYN_STEPS, "step count must be between 1 and 5000.")

    # Aceptar tanto los símbolos griegos (α) como sus nombres latinos (alpha).
    _GREEK_NAMES = {
        "alpha": "α", "beta": "β", "gamma": "γ",
        "epsilon": "ε", "mu": "μ", "delta": "δ",
    }
    for latin, greek in _GREEK_NAMES.items():
        eq_x = re.sub(rf"\b{latin}\b", greek, eq_x)
        eq_y = re.sub(rf"\b{latin}\b", greek, eq_y)

    gate = ("x", "y", "α", "β", "γ", "ε", "μ", "δ")
    compiler.validate(eq_x, variables=gate)
    compiler.validate(eq_y, variables=gate)
    payload = {
        "eq_x": eq_x, "eq_y": eq_y,
        "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon, "mu": mu, "delta": delta,
        "x0": x0, "y0": y0, "t0": t0, "t_fin": t_fin, "h": h,
    }
    result = dynamic_2d_lanchester.Dynamic2DLanchesterService.solve(payload)
    return _finalize_deep(result)


@mcp.tool()
def dynamic_2d_nonhomogeneous_solve(
    a: float = 0.0,
    b: float = -1.0,
    c: float = -9.0,
    d: float = 0.0,
    e: str | float = 1.0,
    f: str | float = 9.0,
    x0: float = 2.0,
    y0: float = 2.0,
    t0: float = 0.0,
    t_fin: float = 5.0,
    h: float = 0.01,
    x_min: float = -5.0,
    x_max: float = 5.0,
    y_min: float = -5.0,
    y_max: float = 5.0,
    cantidad_trayectorias: int = 16,
) -> dict:
    """Solve the non-homogeneous 2D system X' = A·X + B(t) with constant or time-varying forcing.

    Args:
        a, b, c, d: entries of A = [[a, b], [c, d]].
        e, f: forcing components B(t); a number or an expression in t (e.g. "sin(2*t)").
        x0, y0, t0, t_fin, h: integration settings.
        x_min..y_max, cantidad_trayectorias: portrait settings.

    Returns:
        Dict with the classification, equilibrium/particular solution, analytic solution and trajectories.
    """
    x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, traj = _clean_dynamic2d_inputs(
        x0, y0, t0, t_fin, h, x_min, x_max, y_min, y_max, cantidad_trayectorias
    )

    def _forcing(value, name):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        text = str(value)
        normalized = compiler.validate(text, variables=("t",))
        return normalized

    e_val = _forcing(e, "e")
    f_val = _forcing(f, "f")

    payload = {
        "a": a, "b": b, "c": c, "d": d, "e": e_val, "f": f_val,
        "x0": x0, "y0": y0, "t0": t0, "t_fin": t_fin, "h": h,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "cantidad_trayectorias": traj,
    }
    result = dynamic_2d_non_homogeneous.Dynamic2DNonHomogeneousService.solve(payload)
    return _finalize_deep(result)


# --------------------------------------------------------------------------
# Root finding (extended)
# --------------------------------------------------------------------------
def _compile_root(func_str: str) -> Callable:
    normalized = compiler.validate(func_str, variables=("x",))
    return root_finding.RootFindingService.compilar_funcion(normalized)


@mcp.tool()
def root_newton_raphson(
    func_str: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    precision: int = 8,
) -> dict:
    """Find a root of f(x) = 0 with the Newton-Raphson method (numeric derivative).

    Args:
        func_str: Math expression in x.
        x0: Initial guess.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.
        precision: Rounding digits.

    Returns:
        Dict with the root, iteration table and convergence flag.
    """
    x0 = _clean_float(x0, "x0")
    tol = _clean_float(tol, "tol", positive=True)
    max_iter = _clean_int(max_iter, "max_iter", minimum=1, maximum=MAX_ITERATIONS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)

    func = _compile_root(func_str)
    result = root_finding.RootFindingService.newton_raphson(
        func, x0, tol=tol, max_iter=max_iter, precision=precision
    )
    return _finalize(result)


@mcp.tool()
def root_punto_fijo(
    g_str: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    precision: int = 8,
) -> dict:
    """Find a fixed point of g(x) = x with the fixed-point iteration method.

    Converges when |g'(x)| < 1 near the root (Lipschitz check is reported).

    Args:
        g_str: Math expression for the iteration function g(x).
        x0: Initial guess.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.
        precision: Rounding digits.

    Returns:
        Dict with the fixed point, the Lipschitz check, iteration table and convergence flag.
    """
    x0 = _clean_float(x0, "x0")
    tol = _clean_float(tol, "tol", positive=True)
    max_iter = _clean_int(max_iter, "max_iter", minimum=1, maximum=MAX_ITERATIONS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)

    func = _compile_root(g_str)
    result = root_finding.RootFindingService.punto_fijo(
        func, x0, tol=tol, max_iter=max_iter, precision=precision
    )
    return _finalize(result)


@mcp.tool()
def root_aitken(
    g_str: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    precision: int = 8,
) -> dict:
    """Accelerate fixed-point iteration with Aitken's delta-squared method.

    Args:
        g_str: Math expression for the iteration function g(x).
        x0: Initial guess.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.
        precision: Rounding digits.

    Returns:
        Dict with the accelerated root, iteration table and convergence flag.
    """
    x0 = _clean_float(x0, "x0")
    tol = _clean_float(tol, "tol", positive=True)
    max_iter = _clean_int(max_iter, "max_iter", minimum=1, maximum=MAX_ITERATIONS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)

    func = _compile_root(g_str)
    result = root_finding.RootFindingService.aitken(
        func, x0, tol=tol, max_iter=max_iter, precision=precision
    )
    return _finalize(result)


@mcp.tool()
def root_comparar(
    func_str: str,
    g_str: str,
    a: float,
    b: float,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    precision: int = 8,
) -> dict:
    """Compare bisection, fixed-point, Newton-Raphson and Aitken on the same problem.

    Args:
        func_str: Math expression for f(x) (used by bisection/Newton).
        g_str: Math expression for g(x) (used by fixed-point/Aitken).
        a, b: Bracket for bisection (f(a) and f(b) must differ in sign).
        x0: Initial guess.
        tol, max_iter, precision: Shared tolerances.

    Returns:
        Dict with per-method results.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    x0 = _clean_float(x0, "x0")
    tol = _clean_float(tol, "tol", positive=True)
    max_iter = _clean_int(max_iter, "max_iter", minimum=1, maximum=MAX_ITERATIONS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(b > a, "b must be greater than a.")

    compiler.validate(func_str, variables=("x",))
    compiler.validate(g_str, variables=("x",))
    result = root_finding.RootFindingService.comparar_metodos(
        func_str, g_str, a, b, x0, tol=tol, max_iter=max_iter, precision=precision
    )
    return _finalize(result)


# --------------------------------------------------------------------------
# Integration (extended)
# --------------------------------------------------------------------------
def _compile_integrand(func_str: str) -> Callable:
    normalized = compiler.validate(func_str, variables=("x",))
    return integration.IntegrationService.compilar_funcion(normalized)


def _integral_bounds(a: float, b: float, n: int, precision: int, epsilon):
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    n = _clean_int(n, "n", minimum=1, maximum=MAX_SUBINTERVALS)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(b > a, "b must be greater than a.")
    if epsilon is not None:
        epsilon = _clean_float(epsilon, "epsilon")
    return a, b, n, precision, epsilon


@mcp.tool()
def integral_rectangulo(
    func_str: str,
    a: float,
    b: float,
    n: int,
    epsilon: float | None = None,
    precision: int = 8,
) -> dict:
    """Approximate ∫ₐᵇ f(x) dx with the composite midpoint (rectangle) rule.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        n: Number of subintervals.
        epsilon: Optional point in [a, b] for the truncation error bound.
        precision: Rounding digits.

    Returns:
        Dict with the integral, error estimates and table.
    """
    a, b, n, precision, epsilon = _integral_bounds(a, b, n, precision, epsilon)
    func = _compile_integrand(func_str)
    result = integration.IntegrationService.rectangulo_compuesto(
        func, a, b, n=n, precision=precision, epsilon=epsilon
    )
    return _finalize(result)


@mcp.tool()
def integral_trapecio(
    func_str: str,
    a: float,
    b: float,
    n: int,
    epsilon: float | None = None,
    precision: int = 8,
) -> dict:
    """Approximate ∫ₐᵇ f(x) dx with the composite trapezoidal rule.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        n: Number of subintervals.
        epsilon: Optional point in [a, b] for the truncation error bound.
        precision: Rounding digits.

    Returns:
        Dict with the integral, error estimates and table.
    """
    a, b, n, precision, epsilon = _integral_bounds(a, b, n, precision, epsilon)
    func = _compile_integrand(func_str)
    result = integration.IntegrationService.trapecio_compuesto(
        func, a, b, n=n, precision=precision, epsilon=epsilon
    )
    return _finalize(result)


@mcp.tool()
def integral_simpson38(
    func_str: str,
    a: float,
    b: float,
    n: int,
    epsilon: float | None = None,
    precision: int = 8,
) -> dict:
    """Approximate ∫ₐᵇ f(x) dx with the composite Simpson 3/8 rule.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        n: Number of subintervals (must be a multiple of 3).
        epsilon: Optional point in [a, b] for the truncation error bound.
        precision: Rounding digits.

    Returns:
        Dict with the integral, error estimates and table.
    """
    a, b, n, precision, epsilon = _integral_bounds(a, b, n, precision, epsilon)
    _require(n % 3 == 0, "n must be a multiple of 3 for Simpson 3/8.")
    func = _compile_integrand(func_str)
    result = integration.IntegrationService.simpson_38_compuesto(
        func, a, b, n=n, precision=precision, epsilon=epsilon
    )
    return _finalize(result)


@mcp.tool()
def integral_comparar(
    func_str: str,
    a: float,
    b: float,
    n: int,
    epsilon: float | None = None,
    precision: int = 8,
) -> dict:
    """Compare rectangle, trapezoid, Simpson 1/3 and Simpson 3/8 on the same integral.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        n: Number of subintervals (Simpson 1/3 needs even, 3/8 needs multiple of 3).
        epsilon, precision: Error-bound and rounding settings.

    Returns:
        Dict with per-method results and success flags.
    """
    a, b, n, precision, epsilon = _integral_bounds(a, b, n, precision, epsilon)
    compiler.validate(func_str, variables=("x",))
    result = integration.IntegrationService.comparar_metodos(
        func_str, a, b, n=n, precision=precision, epsilon=epsilon
    )
    return _finalize(result)


# --------------------------------------------------------------------------
# ODE (Euler / Heun) and Monte Carlo convergence
# --------------------------------------------------------------------------
def _ode_solve(metodo: str, ecuacion_str, x0, y0, xf, h, tol, precision) -> dict:
    x0 = _clean_float(x0, "x0")
    y0 = _clean_float(y0, "y0")
    xf = _clean_float(xf, "xf")
    h = _clean_float(h, "h", positive=True)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(xf > x0, "xf must be greater than x0.")
    if tol is not None:
        tol = _clean_float(tol, "tol", positive=True)

    n_steps = round((xf - x0) / h)
    _require(1 <= n_steps <= MAX_ODE_STEPS, "step count must be between 1 and 2000.")

    normalized = compiler.validate(ecuacion_str, variables=("x", "y"))
    result = ode.ODEService.ejecutar_metodo(
        metodo, normalized, x0, y0, xf, h, precision=precision, tol=tol
    )
    return _finalize(result, downsample_keys=("x_plot", "y_plot", "y_exacta_plot"))


@mcp.tool()
def ode_euler(
    ecuacion_str: str,
    x0: float,
    y0: float,
    xf: float,
    h: float,
    tol: float | None = None,
    precision: int = 8,
) -> dict:
    """Solve y'(x) = f(x, y), y(x0) = y0 with the explicit Euler method (1st order).

    Args:
        ecuacion_str: Right-hand side f(x, y).
        x0, y0: Initial condition.
        xf: Final x (must be > x0).
        h: Step size (must be positive).
        tol: Optional tolerance flag for the final error.
        precision: Rounding digits.

    Returns:
        Dict with the numerical solution, exact solution (when available) and errors.
    """
    return _ode_solve("euler", ecuacion_str, x0, y0, xf, h, tol, precision)


@mcp.tool()
def ode_heun(
    ecuacion_str: str,
    x0: float,
    y0: float,
    xf: float,
    h: float,
    tol: float | None = None,
    precision: int = 8,
) -> dict:
    """Solve y'(x) = f(x, y), y(x0) = y0 with Heun's predictor-corrector method (2nd order).

    Args:
        ecuacion_str: Right-hand side f(x, y).
        x0, y0: Initial condition.
        xf: Final x (must be > x0).
        h: Step size (must be positive).
        tol: Optional tolerance flag for the final error.
        precision: Rounding digits.

    Returns:
        Dict with the numerical solution, exact solution (when available) and errors.
    """
    return _ode_solve("heun", ecuacion_str, x0, y0, xf, h, tol, precision)


@mcp.tool()
def mc_convergencia_1d(
    func_str: str,
    a: float,
    b: float,
    N: int = 10000,
    seed: int | None = None,
    precision: int = 8,
) -> dict:
    """Show how the mean-value Monte Carlo estimate converges as samples accumulate.

    Args:
        func_str: Math expression in x.
        a, b: Integration limits (b > a).
        N: Total number of samples.
        seed: Optional RNG seed.
        precision: Rounding digits.

    Returns:
        Dict with the cumulative running average over the samples.
    """
    a = _clean_float(a, "a")
    b = _clean_float(b, "b")
    N = _clean_int(N, "N", minimum=1, maximum=MAX_MC_N)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(b > a, "b must be greater than a.")
    if seed is not None:
        seed = _clean_int(seed, "seed", minimum=0, maximum=2**31 - 1)

    normalized = compiler.validate(func_str, variables=("x",))
    func = monte_carlo.MonteCarloService.compilar_funcion(normalized, variables="x")
    result = monte_carlo.MonteCarloService.convergencia_1d(
        func, a, b, N=N, seed=seed, precision=precision
    )
    return _finalize_deep(result)


# --------------------------------------------------------------------------
# Differentiation
# --------------------------------------------------------------------------
@mcp.tool()
def finite_differences(
    func_str: str,
    x_val: float,
    h: float = 1e-5,
    precision: int = 8,
) -> dict:
    """Approximate the first and second derivatives of f(x) at x with finite differences.

    Computes forward, backward and central differences for the first derivative,
    plus the central second derivative, and compares each against the exact
    derivative from SymPy.

    Args:
        func_str: Math expression in x.
        x_val: Point where the derivative is evaluated.
        h: Step size (small, positive).
        precision: Rounding digits.

    Returns:
        Dict with the exact and numerical derivatives and their errors.
    """
    x_val = _clean_float(x_val, "x_val")
    h = _clean_float(h, "h", positive=True)
    precision = _clean_int(precision, "precision", minimum=0, maximum=MAX_PRECISION)
    _require(h < 1.0, "h should be small (typically <= 1e-3).")

    normalized = compiler.validate(func_str, variables=("x",))
    result = differentiation.DifferentiationService.calcular_diferencias_completas(
        normalized, x_val=x_val, h=h, precision=precision
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
