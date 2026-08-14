import numpy as np
from typing import Callable, Tuple, List, Dict, Any
import sympy as sp
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ==========================================================================
# SECURE EXPRESSION PARSING
# --------------------------------------------------------------------------
# SymPy's ``sympify`` and ``parse_expr`` evaluate strings with ``eval`` under
# the hood and allow attribute access and ``__import__`` on recent versions
# (verified: sympify("__import__('os').system(...)") executes the command).
# Every user-supplied expression must go through ``safe_sympify`` /
# ``safe_parse_expr`` which run a strict lexical gate BEFORE parsing:
#   * length cap (denial-of-service mitigation)
#   * identifier whitelist (variables + known functions + constants)
#   * no "." or "__" (blocks attribute access)
#   * integer-digit cap (blocks memory-exhausting eager evaluation)
# ==========================================================================

MAX_EXPRESSION_LENGTH = 200
MAX_INTEGER_DIGITS = 6

ALLOWED_FUNCTIONS = frozenset(
    {
        # trigonometric
        "sin", "cos", "tan", "cot", "sec", "csc",
        "asin", "acos", "atan", "acot", "asec", "acsc", "atan2",
        "sinh", "cosh", "tanh", "coth", "sech", "csch",
        "asinh", "acosh", "atanh",
        # exponential / logarithmic / roots
        "exp", "log", "ln", "sqrt", "cbrt", "root",
        # absolute value / sign / rounding
        "Abs", "sign", "floor", "ceiling", "frac",
        # gamma family
        "gamma", "loggamma", "factorial", "binomial", "subfactorial",
        # special functions
        "erf", "erfc", "erfi",
        "Min", "Max",
    }
)

_LOGIC_KEYWORDS = frozenset({"and", "or", "not"})
_CONSTANTS = frozenset({"e", "E", "pi"})

_IDENTIFIER_RE = re.compile(r"[^\W\d_]\w*", re.UNICODE)
_NUMBER_RE = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_SINGLE_VAR = re.compile(r"(?<![A-Za-z0-9_])[eE]\^")
_ALLOWED_PUNCTUATION = set("()+-,*/%<>!=:")


def normalize_expr(expr: str) -> str:
    """Normalize common math notations to SymPy syntax.

    ``e^x`` / ``e^(-x)`` become ``E**x`` / ``E**(-x)`` (fixes the old
    unbalanced ``exp(`` replacement), ``sen`` becomes ``sin`` and every
    remaining ``^`` becomes ``**``.
    """
    out = expr.replace("sen", "sin")
    out = _SINGLE_VAR.sub("E**", out)
    out = out.replace("^", "**")
    return out


def _check_lexical(expr: str, allowed_identifiers) -> None:
    """Reject any token that is not a number, allowed variable/constant or
    allowed function, and any attribute access (``.`` / ``__``)."""
    allowed_words = (
        set(allowed_identifiers)
        | ALLOWED_FUNCTIONS
        | _CONSTANTS
        | _LOGIC_KEYWORDS
    )

    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue

        number = _NUMBER_RE.match(expr, i)
        if number:
            text = number.group()
            int_part = text.split(".")[0].split("e")[0].split("E")[0].lstrip("+-")
            if len(int_part) > MAX_INTEGER_DIGITS:
                raise ValueError(
                    f"Integer literal too large ({len(int_part)} digits, max {MAX_INTEGER_DIGITS})."
                )
            i = number.end()
            continue

        ident = _IDENTIFIER_RE.match(expr, i)
        if ident:
            token = ident.group()
            if token not in allowed_words:
                raise ValueError(f"Unknown or forbidden token: {token}")
            i = ident.end()
            continue

        if ch in _ALLOWED_PUNCTUATION:
            i += 1
            continue
        if ch == ".":
            raise ValueError("Attribute access is not allowed.")
        if ch == "_":
            raise ValueError("Underscore identifiers are not allowed.")
        raise ValueError(f"Invalid character: {ch!r}")


def safe_sympify(expr_str: str, local_dict: Dict = None, allowed_symbols=()) -> sp.Expr:
    """Lexically vetted ``sp.sympify``. ``allowed_symbols`` lists the variable
    names the expression may reference (plus the constants e/pi/E)."""
    if not expr_str or not expr_str.strip():
        raise ValueError("Expression cannot be empty.")
    if len(expr_str) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression too long ({len(expr_str)} chars, max {MAX_EXPRESSION_LENGTH})."
        )
    normalized = normalize_expr(expr_str)
    _check_lexical(normalized, allowed_symbols)
    try:
        return sp.sympify(normalized, locals=local_dict)
    except Exception as e:
        raise ValueError(f"Invalid math expression: {str(e)}") from e


def safe_parse_expr(expr_str: str, local_dict: Dict = None, allowed_symbols=(),
                    transformations: Tuple = ()) -> sp.Expr:
    """Lexically vetted ``parse_expr``. ``allowed_symbols`` lists the variable
    names the expression may reference (plus the constants e/pi/E)."""
    if not expr_str or not expr_str.strip():
        raise ValueError("Expression cannot be empty.")
    if len(expr_str) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression too long ({len(expr_str)} chars, max {MAX_EXPRESSION_LENGTH})."
        )
    normalized = normalize_expr(expr_str)
    _check_lexical(normalized, allowed_symbols)
    try:
        return parse_expr(normalized, local_dict=local_dict, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Invalid math expression: {str(e)}") from e


# ==========================================================================
# Legacy helpers (kept for backwards compatibility)
# ==========================================================================

def parse_function(func_str: str) -> Callable:
    """
    Convierte string a función vectorizada NumPy.
    Ejemplo: "sin(x) + x**2" -> función callable
    """
    try:
        x = sp.Symbol('x')
        expr = safe_parse_expr(
            func_str,
            local_dict={'e': sp.E, 'pi': sp.pi},
            allowed_symbols=['x'],
            transformations=(standard_transformations + (implicit_multiplication_application,))
        )
        f_numpy = sp.lambdify(x, expr, modules=['numpy'])
        return f_numpy
    except Exception as e:
        raise ValueError(f"Error al parsear función: {str(e)}")


def numerical_derivative(f: Callable, x: float, order: int = 1, h: float = 1e-6) -> float:
    """
    Calcula derivada numérica usando diferencias centrales.
    order: 1 (primera derivada), 2 (segunda), 3 (tercera), 4 (cuarta)
    """
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    elif order == 3:
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)
    elif order == 4:
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)
    else:
        raise ValueError("Order debe ser 1, 2, 3 o 4")


def generate_x_values(a: float, b: float, n: int = 100) -> np.ndarray:
    """Genera n valores equidistantes entre a y b"""
    return np.linspace(a, b, n)


def safe_eval(f: Callable, x: float | np.ndarray) -> float | np.ndarray:
    """Evalúa función de forma segura, manejando excepciones"""
    try:
        result = f(x)
        if np.isnan(result) or np.isinf(result):
            return None
        return result
    except:
        return None


def format_number(value: float, precision: int = 6) -> str:
    """Formatea número con precisión específica"""
    if value is None or np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value:.{precision}f}"


def create_plotly_json(x_data: List, y_data: List, title: str, x_label: str = "x",
                       y_label: str = "y", mode: str = "lines") -> Dict[str, Any]:
    """Crea JSON compatible con Plotly"""
    return {
        "data": [{
            "x": x_data,
            "y": y_data,
            "mode": mode,
            "type": "scatter",
            "name": title
        }],
        "layout": {
            "title": title,
            "xaxis": {"title": x_label},
            "yaxis": {"title": y_label},
            "hovermode": "closest"
        }
    }
