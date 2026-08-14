"""Hardened math expression validation for the MCP server.

The vendored math core trusts the strings it receives. Since MCP tools are
driven by an LLM (which can be tricked by prompt injection), this module adds
defense in depth before any string reaches the math core:

- Length cap (mitigates denial-of-service on pathological inputs)
- Symbol whitelist (no undeclared variables)
- Function whitelist (only standard math functions)
- Notation normalization (``e^x`` -> ``E**x``, ``sen`` -> ``sin``, ``^`` -> ``**``)
- Strict lexical check BEFORE parsing (this is what stops code execution)

**Important:** SymPy's ``sympify`` evaluates strings with ``eval`` internally
and, on some versions, allows attribute access and ``__import__`` (verified:
``sympify("__import__('os').system('echo HACK')")`` executes the command).
Therefore the lexical whitelist below runs FIRST and rejects any token that is
not a number, an allowed variable/constant or an allowed function name, as well
as any ``.`` or ``__``. Only after that gate is it safe to call ``sympify``.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

import sympy as sp

MAX_EXPRESSION_LENGTH = 200

# Reject integer literals with more than this many digits to keep sympify's
# eager evaluation (e.g. factorial(99999999)) from exhausting memory.
MAX_INTEGER_DIGITS = 6

# SymPy canonical function names allowed inside expressions.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
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

# Numeric constants always available regardless of the allowed symbol list.
_BUILTIN_CONSTANTS = {"e": sp.E, "pi": sp.pi, "E": sp.E}

_SINGLE_VAR = re.compile(r"(?<![A-Za-z0-9_])[eE]\^")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_ALLOWED_PUNCTUATION = set("()+-,*/%<>!=:")

# Logic keywords SymPy understands; harmless in expressions.
_LOGIC_KEYWORDS = frozenset({"and", "or", "not"})


def normalize(expr: str) -> str:
    """Normalize common math notations to SymPy syntax.

    Converts ``e^x`` / ``e^(-x)`` into ``E**x`` / ``E**(-x)`` (the old
    ``exp(`` replacement left unbalanced parentheses), ``sen`` into ``sin``,
    ``ln`` into ``log`` and every remaining ``^`` into ``**``.
    """
    out = expr.replace("sen", "sin")
    out = out.replace("ln", "log")
    out = _SINGLE_VAR.sub("E**", out)
    out = out.replace("^", "**")
    return out


def _check_lexical(expr: str, allowed_identifiers: set[str]) -> None:
    """Reject any token that is not a number, variable, constant or allowed
    function, and any attribute access (``.`` / ``__``).

    Runs *before* ``sympify`` so arbitrary Python code can never be evaluated.
    """
    allowed_words = (
        allowed_identifiers
        | set(ALLOWED_FUNCTIONS)
        | set(_BUILTIN_CONSTANTS)
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


def validate(expr_str: str, variables: Sequence[str] = ("x",)) -> str:
    """Validate a math expression string and return its normalized form.

    Raises ``ValueError`` if the expression is too long, malformed, uses
    forbidden functions or references symbols outside ``variables``.
    """
    if not expr_str or not expr_str.strip():
        raise ValueError("Expression cannot be empty.")
    if len(expr_str) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression too long ({len(expr_str)} chars, max {MAX_EXPRESSION_LENGTH})."
        )

    normalized = normalize(expr_str)

    _check_lexical(normalized, {str(name) for name in variables})

    local_dict: dict[str, sp.Symbol] = {}
    for name in variables:
        local_dict[str(name)] = sp.Symbol(str(name), real=True)
    local_dict.update(_BUILTIN_CONSTANTS)

    try:
        expr = sp.sympify(normalized, locals=local_dict)
    except Exception as exc:  # SymPy raises a variety of parse errors
        raise ValueError(f"Invalid math expression: {exc}") from exc

    allowed_symbols = {str(name) for name in variables}
    extra = {str(sym) for sym in expr.free_symbols} - allowed_symbols
    if extra:
        raise ValueError(f"Unknown symbol(s): {sorted(extra)}")

    for fn in expr.atoms(sp.Function):
        name = fn.func.__name__
        if name not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function not allowed: {name}")

    return normalized


def compile_callable(
    expr_str: str,
    variables: Sequence[str] = ("x",),
    modules: Sequence[str] = ("numpy",),
) -> Callable:
    """Validate an expression and return a vectorized NumPy callable.

    The returned callable accepts the variable(s) in the same order as
    ``variables`` (a single callable for one variable, multiple positional
    arguments for several variables).
    """
    normalized = validate(expr_str, variables)
    symbols = [sp.Symbol(str(name), real=True) for name in variables]
    local_dict = {str(name): sym for name, sym in zip(variables, symbols)}
    local_dict.update(_BUILTIN_CONSTANTS)
    expr = sp.sympify(normalized, locals=local_dict)
    return sp.lambdify(symbols, expr, modules=list(modules))


def sanitize_and_check(
    expr_str: str, variables: Iterable[str] = ("x",)
) -> tuple[str, sp.Expr]:
    """Return ``(normalized_str, parsed_expr)`` after validation.

    Useful for tests and for callers that need the symbolic expression too.
    """
    normalized = validate(expr_str, list(variables))
    symbols = [sp.Symbol(str(name), real=True) for name in variables]
    local_dict = {str(name): sym for name, sym in zip(variables, symbols)}
    local_dict.update(_BUILTIN_CONSTANTS)
    expr = sp.sympify(normalized, locals=local_dict)
    return normalized, expr
