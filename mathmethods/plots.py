"""Optional visual output (charts) for the MathematicalMethods MCP server.

This module renders the results of the numerical tools as PNG images that can
be attached to an MCP tool response as ``ImageContent`` (see
``mathmethods.server._maybe_plot``).

Design notes
------------
* **Matplotlib is an optional dependency** (extra ``viz``). It is imported
  lazily inside :func:`render` so that the stdio server startup stays fast
  when the extra is not installed. Missing matplotlib raises a clear
  ``RuntimeError`` telling the user how to install it.
* We use ``Figure`` + ``FigureCanvasAgg`` instead of ``pyplot``: no global
  state, no figure manager, thread-safe and headless.
* Plotters receive ``(fig, result, args)`` where ``result`` is the already
  finalized, JSON-safe dict returned by the tools (NaN -> ``None``) and
  ``args`` carries the original validated inputs for the few plots that need
  to re-evaluate the function (integrals, roots, interpolation, ...).
* Every plotter must tolerate ``None`` and empty lists.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import numpy as np

# Tool name -> plotter. Populated through the ``@register`` decorator.
PLOTTERS: dict[str, Callable[[Any, dict, dict], None]] = {}


def register(*names: str) -> Callable[[Callable], Callable]:
    """Register the decorated function as the plotter for ``names``."""

    def decorator(fn: Callable[[Any, dict, dict], None]) -> Callable:
        for name in names:
            PLOTTERS[name] = fn
        return fn

    return decorator


# --------------------------------------------------------------------------
# Rendering plumbing (lazy matplotlib)
# --------------------------------------------------------------------------
def _figure(figsize: tuple[float, float] = (8.0, 4.5)):
    """Create a backend-agnostic Agg figure.

    Raises ``RuntimeError`` (with install instructions) when matplotlib is not
    available, so the tools can surface an actionable message.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch/subprocess
        raise RuntimeError(
            "Plots require the 'viz' extra: pip install \"mathmethods-mcp[viz]\" "
            "(or `uv sync --extra viz` from a checkout)."
        ) from exc
    fig = Figure(figsize=figsize, dpi=100)
    FigureCanvasAgg(fig)
    return fig


def render(tool: str, result: dict, args: dict | None = None) -> bytes:
    """Render the result of ``tool`` to PNG bytes."""
    fn = PLOTTERS.get(tool)
    if fn is None:
        raise ValueError(f"No plotter registered for tool {tool!r}.")
    fig = _figure()
    fn(fig, result, args or {})
    buf = io.BytesIO()
    # metadata Software=None keeps the PNG deterministic across renders.
    fig.savefig(buf, format="png", metadata={"Software": None})
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _arr(values: Any) -> np.ndarray:
    """Convert a JSON-safe list (possibly containing ``None``) to a float array."""
    if not values:
        return np.array([], dtype=float)
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _sample(expr: str, lo: float, hi: float, n: int = 400):
    """Compile ``expr`` in ``x`` and sample it safely on ``[lo, hi]``.

    Some whitelisted functions (``gamma``, ``erf``, ``factorial``, ...) are
    mapped by SymPy to *scalar* callables, so calling them with an array raises.
    We fall back to a per-point evaluation in that case.
    """
    from mathmethods import compiler

    func = compiler.compile_callable(expr, variables=("x",))
    xs = np.linspace(lo, hi, n)
    with np.errstate(all="ignore"):
        try:
            ys = np.asarray(func(xs), dtype=float)
        except Exception:
            ys = np.array([_eval_point(func, xi) for xi in xs], dtype=float)
    if ys.ndim == 0:
        ys = np.full_like(xs, float(ys), dtype=float)
    ys = ys.copy()
    ys[~np.isfinite(ys)] = np.nan
    return xs, ys


def _eval_point(func, xi: float) -> float:
    """Evaluate a (possibly scalar-only) callable at a single point."""
    try:
        with np.errstate(all="ignore"):
            return float(func(xi))
    except Exception:
        return float("nan")


def _title(ax, text: str) -> None:
    ax.set_title(text, fontsize=10)


# --------------------------------------------------------------------------
# ODE
# --------------------------------------------------------------------------
@register("ode_rk4", "ode_heun", "ode_euler")
def _plot_ode(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    x = _arr(result.get("x_plot"))
    y = _arr(result.get("y_plot"))
    ax.plot(x, y, color="C0", label="numérica")
    exact = result.get("y_exacta_plot") or []
    if len(exact) and len(_arr(exact)) == len(x):
        ax.plot(x, _arr(exact), "--", color="C1", label="exacta")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _title(ax, f"EDO — {args.get('metodo', '')} (h={result.get('h')})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


# --------------------------------------------------------------------------
# Dynamic 1D
# --------------------------------------------------------------------------
@register("dynamic_1d_solve", "dynamic_1d_equilibria")
def _plot_dynamic_1d(fig, result: dict, args: dict) -> None:
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    phase = result.get("phase") or {}
    x = _arr(phase.get("x"))
    fx = _arr(phase.get("fx"))
    ax1.plot(x, fx, color="C0")
    ax1.axhline(0, color="k", lw=0.8)
    for eq in result.get("equilibria") or []:
        xi = eq.get("x")
        if xi is None:
            continue
        color = {"estable": "green", "inestable": "red"}.get(eq.get("stability"), "gray")
        ax1.plot([xi], [0], "o", color=color, ms=6)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    _title(ax1, "f(x) y equilibrios")
    ax1.grid(True, alpha=0.3)

    time = result.get("time") or {}
    t = _arr(time.get("t"))
    for series in time.get("series") or []:
        y = _arr(series.get("x"))
        if len(y) == len(t) and len(y):
            ax2.plot(t, y, label=f"x0={series.get('x0')}")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x(t)")
    _title(ax2, "Series temporales")
    ax2.grid(True, alpha=0.3)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=6)


@register("dynamic_1d_bifurcation")
def _plot_bifurcation(fig, result: dict, args: dict) -> None:
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    bif = result.get("bifurcation") or {}
    rows = bif.get("equilibria") or []
    for stability, color in (("estable", "green"), ("inestable", "red")):
        pts = [
            (r.get("param"), r.get("x"))
            for r in rows
            if r.get("stability") == stability and r.get("param") is not None and r.get("x") is not None
        ]
        if pts:
            px, py = zip(*pts, strict=False)
            ax1.scatter(px, py, s=5, color=color, label=stability)
    ax1.set_xlabel(result.get("bif_param", "parámetro"))
    ax1.set_ylabel("x")
    _title(ax1, "Diagrama de bifurcación")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    for sl in result.get("phase_slices") or []:
        ph = sl.get("phase") or {}
        if ph.get("x"):
            ax2.plot(_arr(ph["x"]), _arr(ph["fx"]), label=f"p={sl.get('param')}")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    _title(ax2, "Cortes de fase")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)


# --------------------------------------------------------------------------
# Dynamic 2D
# --------------------------------------------------------------------------
@register(
    "dynamic_2d_linear_solve",
    "dynamic_2d_nonlinear_solve",
    "dynamic_2d_conservative_solve",
    "dynamic_2d_lanchester_solve",
    "dynamic_2d_nonhomogeneous_solve",
)
def _plot_dynamic_2d(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)

    for traj in result.get("automatic_trajectories") or []:
        x = _arr(traj.get("x"))
        y = _arr(traj.get("y"))
        if len(x) and len(x) == len(y):
            ax.plot(x, y, color="0.75", lw=0.8, zorder=1)

    principal = result.get("principal_trajectory") or {}
    px = _arr(principal.get("x"))
    py = _arr(principal.get("y"))
    if len(px) and len(px) == len(py):
        ax.plot(px, py, color="C0", lw=1.8, zorder=3, label="principal")

    for eq in result.get("equilibrio"),:
        if isinstance(eq, dict) and eq.get("x") is not None and eq.get("y") is not None:
            ax.plot([eq["x"]], [eq["y"]], "ko", ms=6, zorder=4, label="equilibrio")

    for pt in result.get("puntos_analizados") or []:
        if pt.get("x") is not None and pt.get("y") is not None:
            ax.plot([pt["x"]], [pt["y"]], "ko", ms=5, zorder=4)

    for key, color in (("nulclina_x", "C2"), ("nulclina_y", "C3")):
        nul = result.get(key)
        if isinstance(nul, dict):
            pts = nul.get("puntos") or []
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], "--", color=color, lw=1.0)

    for line in result.get("real_eigenvectors_lines") or []:
        if line and len(line) == 2:
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], ":", color="C4", lw=1.0)

    # Nullclines from a mesh grid (nonlinear systems expose `contour_data`).
    contour = result.get("contour_data") or {}
    x_axis = contour.get("x_axis")
    y_axis = contour.get("y_axis")
    if x_axis and y_axis:
        for key, color in (("z_x_nul", "C2"), ("z_y_nul", "C3")):
            z = contour.get(key)
            if not z:
                continue
            try:
                grid = np.asarray(z, dtype=float)
                gx = np.asarray(x_axis, dtype=float)
                gy = np.asarray(y_axis, dtype=float)
                if grid.shape == (len(gy), len(gx)):
                    ax.contour(gx, gy, grid, levels=[0.0], colors=[color], linewidths=1.0)
            except Exception:
                pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _title(ax, "Retrato de fase")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
@register("mc_convergencia_1d")
def _plot_mc_convergence(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    hist = result.get("historial_convergencia") or []
    if hist:
        ns = [h.get("N") for h in hist]
        vals = [h.get("integral") for h in hist]
        ax.plot(ns, vals, color="C0")
    exact = result.get("valor_exacto_gauss")
    if exact is not None:
        ax.axhline(exact, color="C1", ls="--", label=f"referencia≈{exact}")
        ax.legend(fontsize=7)
    ax.set_xlabel("N (muestras)")
    ax.set_ylabel("integral acumulada")
    _title(ax, "Convergencia Monte Carlo")
    ax.grid(True, alpha=0.3)


@register("mc_hit_or_miss_1d")
def _plot_hit_or_miss(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    x_line = _arr(result.get("x_line"))
    y_line = _arr(result.get("y_line"))
    if len(x_line):
        ax.plot(x_line, y_line, color="C0", lw=1.5, label="f(x)", zorder=3)
    hist = result.get("historial") or []
    if hist:
        hx = [h.get("x") for h in hist]
        hy = [h.get("y_rand") for h in hist]
        colors = ["green" if h.get("exito") == "✓" else "red" for h in hist]
        ax.scatter(hx, hy, s=4, c=colors, alpha=0.35, zorder=2)
    lim = result.get("grafico_limites") or {}
    if lim.get("y_base") is not None:
        ax.axhline(lim["y_base"], color="0.6", lw=0.8, ls=":")
    if lim.get("y_techo") is not None:
        ax.axhline(lim["y_techo"], color="0.6", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _title(ax, "Hit-or-Miss")
    ax.grid(True, alpha=0.3)


@register("mc_valor_promedio_1d")
def _plot_mc_mean_1d(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    hist = result.get("historial") or []
    if hist:
        ax.scatter(
            [h.get("x") for h in hist],
            [h.get("f_x") for h in hist],
            s=5,
            alpha=0.4,
            color="C0",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    _title(ax, f"Valor promedio 1D (∫≈{result.get('integral')})")
    ax.grid(True, alpha=0.3)


@register("mc_valor_promedio_2d")
def _plot_mc_mean_2d(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    hist = result.get("historial") or []
    if hist:
        sc = ax.scatter(
            [h.get("x") for h in hist],
            [h.get("y") for h in hist],
            c=[h.get("f_xy") for h in hist],
            s=5,
            cmap="viridis",
            alpha=0.6,
        )
        fig.colorbar(sc, ax=ax, label="f(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _title(ax, f"Valor promedio 2D (∫≈{result.get('integral')})")


@register("mc_valor_promedio_3d")
def _plot_mc_mean_3d(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    hist = result.get("historial") or []
    values = [h.get("f_xyz") for h in hist if h.get("f_xyz") is not None]
    if values:
        ax.hist(values, bins=30, color="C0", alpha=0.8)
    ax.set_xlabel("f(x,y,z)")
    ax.set_ylabel("frecuencia")
    _title(ax, f"Valor promedio 3D (∫≈{result.get('integral')})")
    ax.grid(True, alpha=0.3)


@register("mc_estadistico_1d")
def _plot_mc_statistical(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    values = result.get("distribucion") or []
    if values:
        ax.hist(values, bins=30, color="C0", alpha=0.8)
    for key, color, label in (
        ("integral", "C1", "media"),
        ("ic_inf", "C2", "IC inf"),
        ("ic_sup", "C2", "IC sup"),
    ):
        val = result.get(key)
        if val is not None:
            ax.axvline(val, color=color, ls="--", lw=1.2, label=label)
    ax.set_xlabel("estimación de la integral")
    ax.set_ylabel("frecuencia")
    _title(ax, "Distribución de estimaciones (M réplicas)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------
@register("integral_rectangulo", "integral_trapecio", "integral_simpson13", "integral_simpson38")
def _plot_integral(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    func_str = args.get("func_str")
    a = args.get("a")
    b = args.get("b")
    if func_str is not None and a is not None and b is not None and b > a:
        xs, ys = _sample(func_str, a, b, n=400)
        ax.plot(xs, ys, color="C0", label="f(x)")
        ax.fill_between(xs, 0, np.nan_to_num(ys), color="C0", alpha=0.2)
        ax.axvline(a, color="0.6", lw=0.8)
        ax.axvline(b, color="0.6", lw=0.8)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    _title(ax, f"{result.get('metodo', 'Integral')} → ∫≈{result.get('integral')}")
    ax.grid(True, alpha=0.3)


@register("integral_comparar")
def _plot_integral_compare(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    comp = result.get("comparativa") or {}
    labels = list(comp.keys())
    values = []
    for label in labels:
        entry = comp.get(label) or {}
        values.append(entry.get("integral") if entry.get("exito") else np.nan)
    xpos = np.arange(len(labels))
    ax.bar(xpos, [0 if v is None else v for v in values], color="C0", alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("integral")
    _title(ax, "Comparativa de reglas de integración")
    ax.grid(True, axis="y", alpha=0.3)


# --------------------------------------------------------------------------
# Root finding
# --------------------------------------------------------------------------
@register("root_bisection")
def _plot_root_bisection(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    func_str = args.get("func_str")
    a = args.get("a")
    b = args.get("b")
    if func_str is not None and a is not None and b is not None and b > a:
        pad = (b - a) * 0.05
        xs, ys = _sample(func_str, a - pad, b + pad, n=400)
        ax.plot(xs, ys, color="C0", label="f(x)")
        ax.plot([a, b], [0, 0], "|", color="0.5", ms=10)
    root = result.get("raiz")
    if root is not None:
        ax.axvline(root, color="C3", ls="--", lw=1.2, label=f"raíz≈{root}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    _title(ax, f"Bisección — {result.get('num_iter')} iter")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


@register("root_newton_raphson")
def _plot_root_newton(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    func_str = args.get("func_str")
    x0 = args.get("x0")
    if func_str is not None and x0 is not None:
        span = max(abs(x0) * 0.5, 1.0)
        xs, ys = _sample(func_str, x0 - span, x0 + span, n=400)
        ax.plot(xs, ys, color="C0", label="f(x)")
    root = result.get("raiz")
    if root is not None:
        ax.axvline(root, color="C3", ls="--", lw=1.2, label=f"raíz≈{root}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    _title(ax, f"Newton–Raphson — {result.get('num_iter')} iter")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


@register("root_punto_fijo", "root_aitken")
def _plot_root_fixed_point(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    g_str = args.get("g_str")
    x0 = args.get("x0")
    if g_str is not None and x0 is not None:
        span = max(abs(x0) * 0.5, 1.0)
        xs, ys = _sample(g_str, x0 - span, x0 + span, n=400)
        ax.plot(xs, ys, color="C0", label="g(x)")
        ax.plot(xs, xs, color="0.6", ls="--", lw=1.0, label="y = x")
    root = result.get("raiz")
    if root is not None:
        ax.plot([root], [root], "o", color="C3", ms=6, label=f"punto fijo≈{root}")
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    _title(ax, f"{result.get('metodo', 'Punto fijo')} — {result.get('num_iter')} iter")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


@register("root_comparar")
def _plot_root_compare(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    comp = result.get("comparativa") or {}
    labels = list(comp.keys())
    iters = [(comp.get(label) or {}).get("num_iter") or 0 for label in labels]
    xpos = np.arange(len(labels))
    ax.bar(xpos, iters, color="C0", alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("iteraciones")
    _title(ax, "Comparativa de métodos de raíces")
    ax.grid(True, axis="y", alpha=0.3)


# --------------------------------------------------------------------------
# Interpolation and differentiation
# --------------------------------------------------------------------------
@register("interpolation_lagrange")
def _plot_lagrange(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    puntos = result.get("puntos") or []
    xs = [p.get("x") for p in puntos if p.get("x") is not None]
    ys = [p.get("y") for p in puntos if p.get("x") is not None]
    polinomio = result.get("polinomio")
    if polinomio and xs:
        lo, hi = min(xs), max(xs)
        pad = (hi - lo) * 0.1 or 1.0
        try:
            curve_x, curve_y = _sample(polinomio, lo - pad, hi + pad, n=400)
            ax.plot(curve_x, curve_y, color="C0", label="P(x)")
        except Exception:
            pass
    if xs:
        ax.plot(xs, ys, "o", color="C1", label="puntos")
    x_eval = result.get("x_eval")
    if x_eval is not None and result.get("P_eval") is not None:
        ax.plot([x_eval], [result["P_eval"]], "r*", ms=10, label=f"P({x_eval})")
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    _title(ax, f"Polinomio de Lagrange (grado {result.get('grado')})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


@register("finite_differences")
def _plot_finite_differences(fig, result: dict, args: dict) -> None:
    ax = fig.add_subplot(111)
    func_str = args.get("func_str")
    x_val = args.get("x_val")
    if func_str is not None and x_val is not None:
        span = max(abs(x_val) * 0.5, 1.0)
        xs, ys = _sample(func_str, x_val - span, x_val + span, n=400)
        ax.plot(xs, ys, color="C0", label="f(x)")
        fx = result.get("f_x")
        deriv = result.get("derivada_exacta")
        if fx is not None and deriv is not None:
            tx = np.linspace(x_val - span * 0.15, x_val + span * 0.15, 50)
            ax.plot(tx, fx + deriv * (tx - x_val), "--", color="C1", label="tangente exacta")
            ax.plot([x_val], [fx], "o", color="C3", ms=6)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    _title(ax, f"Diferencias finitas en x={result.get('x_evaluado')}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
