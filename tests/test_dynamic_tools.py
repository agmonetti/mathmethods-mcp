"""Tests for the dynamic-system and Monte Carlo MCP tools."""

import json
import math

import pytest

from mathmethods.server import (
    dynamic_1d_bifurcation,
    dynamic_1d_equilibria,
    dynamic_1d_solve,
    dynamic_2d_conservative_solve,
    dynamic_2d_lanchester_solve,
    dynamic_2d_linear_solve,
    dynamic_2d_nonhomogeneous_solve,
    dynamic_2d_nonlinear_solve,
    mc_estadistico_1d,
    mc_hit_or_miss_1d,
    mc_valor_promedio_1d,
    mc_valor_promedio_2d,
    mc_valor_promedio_3d,
)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def test_mc_hit_or_miss_positive():
    res = mc_hit_or_miss_1d("x", a=0, b=2, N=20000, seed=42)
    assert res["integral"] == pytest.approx(2.0, abs=0.15)


def test_mc_hit_or_miss_signed():
    # sin(x) en [0, 2pi] tiene integral exacta 0 (el viejo bug daba ~4)
    res = mc_hit_or_miss_1d("sin(x)", a=0, b=2 * math.pi, N=40000, seed=42)
    assert res["integral"] == pytest.approx(0.0, abs=0.2)


def test_mc_valor_promedio_1d():
    res = mc_valor_promedio_1d("x**2", a=0, b=2, N=20000, seed=42)
    assert res["integral"] == pytest.approx(8 / 3, abs=0.1)


def test_mc_valor_promedio_2d():
    res = mc_valor_promedio_2d("x + y", x_a=0, x_b=1, y_a=0, y_b=1, N=20000, seed=7)
    assert res["integral"] == pytest.approx(1.0, abs=0.1)


def test_mc_valor_promedio_3d():
    res = mc_valor_promedio_3d("x + y + z", 0, 1, 0, 1, 0, 1, N=20000, seed=7)
    assert res["integral"] == pytest.approx(1.5, abs=0.15)


def test_mc_estadistico():
    res = mc_estadistico_1d("x**2", a=0, b=2, N=500, M=30, seed=42)
    assert res["integral"] == pytest.approx(8 / 3, abs=0.5)
    assert res["ic_inf"] <= res["integral"] <= res["ic_sup"]


# --------------------------------------------------------------------------
# Dynamic 1D
# --------------------------------------------------------------------------
def test_dynamic_1d_solve_verhulst():
    res = dynamic_1d_solve(model="verhulst", params={"mu": 1.0, "K": 2.0}, initial_conditions=[0.5])
    roots = sorted(e["x"] for e in res["equilibria"])
    assert roots == pytest.approx([0.0, 2.0], abs=0.1)
    stabilities = {e["stability"] for e in res["equilibria"]}
    assert "estable" in stabilities and "inestable" in stabilities


def test_dynamic_1d_custom_rce_rejected():
    with pytest.raises(Exception):
        dynamic_1d_solve(func_str="__import__('os').system('true')", model="custom")


def test_dynamic_1d_bifurcation_verhulst():
    res = dynamic_1d_bifurcation(
        model="verhulst", params={"K": 2.0}, bif_param="mu", bif_min=0, bif_max=3, bif_steps=12
    )
    assert len(res["bifurcation"]["equilibria"]) > 0
    assert len(res["phase_slices"]) == 3


def test_dynamic_1d_equilibria():
    res = dynamic_1d_equilibria(func_str="x**2 - 4", model="custom")
    assert res["equilibria"]


# --------------------------------------------------------------------------
# Dynamic 2D
# --------------------------------------------------------------------------
def test_dynamic_2d_linear():
    res = dynamic_2d_linear_solve(cantidad_trayectorias=4)
    valid = (
        "Nodo inestable",
        "Nodo estable",
        "Centro",
        "Silla",
        "Foco inestable",
        "Foco estable",
    )
    assert res["clasificacion"] in valid


def test_dynamic_2d_nonlinear():
    res = dynamic_2d_nonlinear_solve(cantidad_trayectorias=4)
    assert "puntos_analizados" in res


def test_dynamic_2d_conservative():
    res = dynamic_2d_conservative_solve(cantidad_trayectorias=4)
    assert res["es_conservativo"] is True


def test_dynamic_2d_lanchester():
    res = dynamic_2d_lanchester_solve()
    assert res["is_classic"] is True
    assert res["winner_analytic"] in ("Ejército X (Rojo)", "Ejército Y (Azul)", "Empate (Aniquilación Mutua)")


def test_dynamic_2d_nonhomogeneous_time_varying():
    res = dynamic_2d_nonhomogeneous_solve(a=1, b=0, c=0, d=-2, e="sin(t)", f="0", cantidad_trayectorias=4)
    assert "solucion_particular_latex" in res


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda: mc_hit_or_miss_1d("x", 0, 2, N=1000, seed=1),
        lambda: mc_valor_promedio_1d("x", 0, 2, N=1000, seed=1),
        lambda: mc_estadistico_1d("x", 0, 2, N=100, M=10, seed=1),
        lambda: dynamic_1d_solve(initial_conditions=[0.5]),
        lambda: dynamic_1d_bifurcation(bif_steps=8),
        lambda: dynamic_2d_linear_solve(cantidad_trayectorias=2),
        lambda: dynamic_2d_nonlinear_solve(cantidad_trayectorias=2),
        lambda: dynamic_2d_conservative_solve(cantidad_trayectorias=2),
        lambda: dynamic_2d_lanchester_solve(),
        lambda: dynamic_2d_nonhomogeneous_solve(cantidad_trayectorias=2),
    ],
)
def test_dynamic_results_json_serializable(call):
    result = call()
    json.dumps(result)
    assert isinstance(result, dict)
