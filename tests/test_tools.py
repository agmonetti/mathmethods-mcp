"""Tests for the MathematicalMethods MCP tools and the hardened compiler."""

import json
import math

import pytest

from mathmethods import compiler
from mathmethods.server import (
    integral_simpson13,
    interpolation_lagrange,
    ode_rk4,
    root_bisection,
)


# --------------------------------------------------------------------------
# compiler: notation normalization
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("e^x", "E**x"),
        ("e^(-x)", "E**(-x)"),
        ("2*e^x + 1", "2*E**x + 1"),
        ("sen(x)", "sin(x)"),
        ("ln(x)", "log(x)"),
        ("x^2 + sin(x)", "x**2 + sin(x)"),
        ("x**2", "x**2"),
    ],
)
def test_compiler_normalize(raw, expected):
    assert compiler.normalize(raw) == expected


def test_compiler_e_power_roundtrip():
    # e^x must compile and evaluate to exp(x)
    fn = compiler.compile_callable("e^x")
    assert fn(1.0) == pytest.approx(math.e)


def test_compiler_validate_accepts_known_symbols():
    assert compiler.validate("sin(x) + cos(x)", ("x",))
    assert compiler.validate("x*y + y**2", ("x", "y"))


def test_compiler_rejects_unknown_symbols():
    with pytest.raises(ValueError, match="Unknown or forbidden token"):
        compiler.validate("x + z", ("x",))


def test_compiler_rejects_forbidden_functions():
    with pytest.raises(ValueError, match="Unknown or forbidden token"):
        compiler.validate("evil_function(x)", ("x",))


def test_compiler_rejects_empty_and_too_long():
    with pytest.raises(ValueError, match="empty"):
        compiler.validate("  ", ("x",))
    with pytest.raises(ValueError, match="too long"):
        compiler.validate("x" * (compiler.MAX_EXPRESSION_LENGTH + 1), ("x",))


def test_compiler_rejects_rce_payloads(tmp_path):
    marker = tmp_path / "pwned"
    payloads = [
        f"__import__('os').system('touch {marker}')",
        "os.system('touch /tmp/pwned')",
        "open('file.txt','w')",
        "lambda x: x",
        "(lambda: 1)()",
        "globals()",
        "x.__class__",
        "__builtins__['eval']('1')",
    ]
    for payload in payloads:
        with pytest.raises(ValueError):
            compiler.validate(payload, ("x",))
    # The payloads must have been rejected before any evaluation happened.
    assert not marker.exists()


# --------------------------------------------------------------------------
# root_bisection
# --------------------------------------------------------------------------
def test_root_bisection_known_root():
    result = root_bisection(func_str="x**2 - 4", a=0, b=5, tol=1e-8, max_iter=200)
    assert result["convergencia"] is True
    assert result["raiz"] == pytest.approx(2.0, abs=1e-5)


def test_root_bisection_requires_sign_change():
    with pytest.raises(Exception):
        root_bisection(func_str="x**2 - 4", a=3, b=5)


def test_root_bisection_rejects_invalid_input():
    with pytest.raises(Exception):
        root_bisection(func_str="x**2 - 4", a=5, b=0)  # b < a


def test_root_bisection_e_notation():
    result = root_bisection(func_str="e^x - 2", a=0, b=1)
    assert result["convergencia"] is True
    assert result["raiz"] == pytest.approx(math.log(2), abs=1e-4)


# --------------------------------------------------------------------------
# integral_simpson13
# --------------------------------------------------------------------------
def test_simpson13_exact_degree3():
    result = integral_simpson13(func_str="x**3", a=0, b=2, n=2)
    assert result["integral"] == pytest.approx(4.0, rel=1e-5)


def test_simpson13_rejects_odd_n():
    with pytest.raises(Exception):
        integral_simpson13(func_str="x**2", a=0, b=2, n=3)


def test_simpson13_handles_e_notation():
    # Integral of e^x on [0,1] is e - 1
    result = integral_simpson13(func_str="e^x", a=0, b=1, n=20)
    assert result["integral"] == pytest.approx(math.e - 1, rel=1e-4)


# --------------------------------------------------------------------------
# ode_rk4
# --------------------------------------------------------------------------
def test_rk4_converges_to_exact():
    result = ode_rk4(ecuacion_str="y", x0=0, y0=1, xf=1, h=0.1)
    y_final = result["y_plot"][-1]
    y_exacta = result["y_exacta_plot"][-1]
    assert y_final == pytest.approx(y_exacta, abs=1e-3)
    assert y_final == pytest.approx(math.e, abs=1e-3)


def test_rk4_works_when_exact_solution_unavailable():
    # Nonlinear RHS: dsolve cannot produce a usable closed form, but the
    # numerical method must still succeed.
    result = ode_rk4(ecuacion_str="sin(x)*y + y**2", x0=0, y0=1, xf=1, h=0.1)
    assert result["solucion_exacta_str"] is None
    assert result["y_exacta_plot"] == []
    assert len(result["tabla"]) == len(result["x_plot"])


def test_rk4_reaches_endpoint_exactly():
    result = ode_rk4(ecuacion_str="y", x0=0, y0=1, xf=1, h=0.3)
    assert result["x_plot"][-1] == pytest.approx(1.0)


def test_rk4_rejects_nonpositive_h():
    with pytest.raises(Exception):
        ode_rk4(ecuacion_str="y", x0=0, y0=1, xf=1, h=0)


# --------------------------------------------------------------------------
# interpolation_lagrange
# --------------------------------------------------------------------------
def test_lagrange_line():
    result = interpolation_lagrange(puntos_x=[0.0, 2.0], puntos_y=[0.0, 4.0], x_eval=1.0)
    assert result["P_eval"] == pytest.approx(2.0)
    assert result["grado"] <= 1


def test_lagrange_from_function():
    result = interpolation_lagrange(
        puntos_x=[0.0, 1.0, 2.0], func_str="x**2", x_eval=1.5
    )
    assert result["P_eval"] == pytest.approx(2.25)


def test_lagrange_rejects_duplicate_x():
    with pytest.raises(Exception):
        interpolation_lagrange(puntos_x=[0.0, 0.0], puntos_y=[1.0, 2.0])


def test_lagrange_requires_data_source():
    with pytest.raises(Exception):
        interpolation_lagrange(puntos_x=[0.0, 1.0])


# --------------------------------------------------------------------------
# Output contract: every tool result must be JSON-serializable
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda: root_bisection(func_str="x**2 - 4", a=0, b=5),
        lambda: integral_simpson13(func_str="sin(x)/x", a=0.0001, b=1, n=10),
        lambda: ode_rk4(ecuacion_str="y", x0=0, y0=1, xf=1, h=0.1),
        lambda: interpolation_lagrange(puntos_x=[0.0, 1.0, 2.0], func_str="x**2", x_eval=1.5),
    ],
)
def test_results_are_json_serializable(call):
    result = call()
    json.dumps(result)  # must not raise
    assert isinstance(result, dict)
