"""Tests for the extended method tools (roots, integration rules, ODE, convergence, finite differences)."""

import json
import math

import pytest

from mathmethods.server import (
    finite_differences,
    integral_comparar,
    integral_rectangulo,
    integral_simpson38,
    integral_trapecio,
    mc_convergencia_1d,
    ode_euler,
    ode_heun,
    root_aitken,
    root_comparar,
    root_newton_raphson,
    root_punto_fijo,
)


# --------------------------------------------------------------------------
# Root finding (extended)
# --------------------------------------------------------------------------
def test_root_newton_raphson():
    res = root_newton_raphson("x**2 - 4", 5.0)
    assert res["convergencia"] is True
    assert res["raiz"] == pytest.approx(2.0, abs=1e-5)


def test_root_punto_fijo():
    res = root_punto_fijo("sqrt(x + 2)", 1.0)
    assert res["convergencia"] is True
    assert res["raiz"] == pytest.approx(2.0, abs=1e-4)


def test_root_aitken_faster_than_punto_fijo():
    pf = root_punto_fijo("sqrt(x + 2)", 1.0, tol=1e-8)
    ak = root_aitken("sqrt(x + 2)", 1.0, tol=1e-8)
    assert ak["raiz"] == pytest.approx(2.0, abs=1e-4)
    assert ak["num_iter"] <= pf["num_iter"]


def test_root_comparar():
    res = root_comparar("x**2 - 4", "sqrt(x + 2)", 0, 5, 1.0)
    assert set(res["comparativa"].keys()) == {
        "biseccion",
        "punto_fijo",
        "newton_raphson",
        "aitken",
    }


# --------------------------------------------------------------------------
# Integration (extended)
# --------------------------------------------------------------------------
def test_integral_rectangulo():
    res = integral_rectangulo("x**2", 0, 1, 100)
    assert res["integral"] == pytest.approx(1 / 3, abs=1e-3)


def test_integral_trapecio():
    res = integral_trapecio("x**2", 0, 1, 100)
    assert res["integral"] == pytest.approx(1 / 3, abs=1e-4)


def test_integral_simpson38_exact_degree3():
    res = integral_simpson38("x**3", 0, 3, 3)
    assert res["integral"] == pytest.approx(20.25, rel=1e-5)


def test_integral_simpson38_requires_multiple_of_3():
    with pytest.raises(Exception):
        integral_simpson38("x**2", 0, 1, 4)


def test_integral_comparar():
    res = integral_comparar("x**2", 0, 1, 6)
    assert set(res["comparativa"].keys()) == {
        "rectangulo",
        "trapecio",
        "simpson_13",
        "simpson_38",
    }
    assert all(v["exito"] for v in res["comparativa"].values())


# --------------------------------------------------------------------------
# ODE (Euler / Heun)
# --------------------------------------------------------------------------
def test_ode_euler_first_order():
    res = ode_euler("y", 0, 1, 1, 0.1)
    assert res["y_plot"][-1] == pytest.approx(math.e, abs=0.2)


def test_ode_heun_second_order():
    res = ode_heun("y", 0, 1, 1, 0.1)
    assert res["y_plot"][-1] == pytest.approx(math.e, abs=0.02)


def test_ode_heun_more_accurate_than_euler():
    e = ode_euler("y", 0, 1, 1, 0.1)["y_plot"][-1]
    h = ode_heun("y", 0, 1, 1, 0.1)["y_plot"][-1]
    assert abs(h - math.e) < abs(e - math.e)


# --------------------------------------------------------------------------
# Monte Carlo convergence + finite differences
# --------------------------------------------------------------------------
def test_mc_convergencia():
    res = mc_convergencia_1d("x", 0, 2, N=2000, seed=1)
    assert len(res["historial_convergencia"]) > 0
    assert res["valor_exacto_gauss"] == pytest.approx(2.0, abs=1e-6)


def test_finite_differences():
    res = finite_differences("x**2", 3.0, h=0.01)
    assert res["derivada_exacta"] == pytest.approx(6.0, abs=1e-9)
    central = next(r for r in res["resultados"] if r["metodo"] == "Diferencia Central")
    assert central["valor"] == pytest.approx(6.0, abs=1e-3)
    # forward/backward are 1st order (less accurate than central)
    prog = next(r for r in res["resultados"] if r["metodo"] == "Diferencia Progresiva")
    assert prog["error_raw"] > central["error_raw"]


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda: root_newton_raphson("x**2 - 4", 5.0),
        lambda: root_punto_fijo("sqrt(x + 2)", 1.0),
        lambda: root_aitken("sqrt(x + 2)", 1.0),
        lambda: root_comparar("x**2 - 4", "sqrt(x + 2)", 0, 5, 1.0),
        lambda: integral_rectangulo("x**2", 0, 1, 10),
        lambda: integral_trapecio("x**2", 0, 1, 10),
        lambda: integral_simpson38("x**3", 0, 3, 3),
        lambda: integral_comparar("x**2", 0, 1, 6),
        lambda: ode_euler("y", 0, 1, 1, 0.1),
        lambda: ode_heun("y", 0, 1, 1, 0.1),
        lambda: mc_convergencia_1d("x", 0, 2, N=100, seed=1),
        lambda: finite_differences("x**2", 3.0, 0.01),
    ],
)
def test_extended_results_json_serializable(call):
    result = call()
    json.dumps(result)
    assert isinstance(result, dict)
