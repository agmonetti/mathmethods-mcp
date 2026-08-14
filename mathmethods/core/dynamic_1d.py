import math
import re
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from .utils import numerical_derivative, safe_parse_expr


class Dynamic1DService:
    @staticmethod
    def _sanitize_expr(expr: str) -> str:
        cleaned = expr.replace('sen', 'sin')
        cleaned = re.sub(r'(?<![A-Za-z0-9_])[eE]\^', 'E**', cleaned).replace('^', '**')
        return cleaned

    @staticmethod
    def _parse_expression(expr_str: str, params: Dict[str, float], allowed_symbols: List[str]) -> sp.Expr:
        expr_str = Dynamic1DService._sanitize_expr(expr_str)
        x = sp.Symbol('x', real=True)

        local_dict: Dict[str, Any] = {
            'x': x,
            'e': sp.E,
            'pi': sp.pi,
        }

        for key in params.keys():
            local_dict[key] = sp.Symbol(key, real=True)

        for sym in allowed_symbols:
            if sym not in local_dict:
                local_dict[sym] = sp.Symbol(sym, real=True)

        gate_allowed = list(dict.fromkeys(list(allowed_symbols) + list(params.keys())))
        expr = safe_parse_expr(
            expr_str,
            local_dict=local_dict,
            allowed_symbols=gate_allowed,
            transformations=(standard_transformations + (implicit_multiplication_application,)),
        )

        free_symbols = {str(sym) for sym in expr.free_symbols}
        allowed = set(allowed_symbols)
        if not free_symbols.issubset(allowed):
            missing = ', '.join(sorted(free_symbols - allowed))
            raise ValueError(f"Faltan parametros: {missing}")

        return expr

    @staticmethod
    def _expr_to_str(expr: sp.Expr) -> str:
        return str(expr).replace('**', '^')

    @staticmethod
    def _format_inequality_result(result: Any) -> str | None:
        if result is None:
            return None
        if result is True:
            return 'siempre'
        if result is False:
            return 'nunca'
        try:
            return Dynamic1DService._expr_to_str(result)
        except Exception:
            return str(result)

    @staticmethod
    def _extract_real_conditions(expr: sp.Expr) -> List[Relational]:
        conditions: List[sp.Relational] = []
        for pow_expr in expr.atoms(sp.Pow):
            exp = pow_expr.exp
            if exp.is_Rational and exp.q % 2 == 0:
                conditions.append(sp.Ge(sp.simplify(pow_expr.base), 0))
        for log_expr in expr.atoms(sp.log):
            if log_expr.args:
                arg = sp.simplify(log_expr.args[0])
                if arg.has(sp.Abs):
                    conditions.append(sp.Ne(arg, 0))
                else:
                    conditions.append(sp.Gt(arg, 0))
        return conditions

    @staticmethod
    def _exact_bifurcation_analysis(expr: sp.Expr, bif_param: str) -> Dict[str, Any]:
        x = sp.Symbol('x', real=True)
        p = sp.Symbol(bif_param, real=True)

        expr_simplified = sp.simplify(expr)
        deriv_expr = sp.diff(expr_simplified, x)

        def extract_finite_roots(roots_set: sp.Set) -> List[sp.Expr] | None:
            if isinstance(roots_set, sp.FiniteSet):
                return list(roots_set)
            if isinstance(roots_set, sp.Intersection):
                for arg in roots_set.args:
                    extracted = extract_finite_roots(arg)
                    if extracted is not None:
                        return extracted
                return None
            if isinstance(roots_set, sp.Union):
                combined: List[sp.Expr] = []
                for arg in roots_set.args:
                    extracted = extract_finite_roots(arg)
                    if extracted is None:
                        return None
                    combined.extend(extracted)
                return combined
            return None

        try:
            roots_set = sp.solveset(sp.Eq(expr_simplified, 0), x, domain=sp.S.Reals)
            roots = extract_finite_roots(roots_set)
            if roots is None:
                raise ValueError('Soluciones no finitas o no reales')
        except Exception:
            roots = []

        if any(root.has(sp.LambertW) for root in roots):
            raise ValueError('Raices con LambertW no soportadas')
            
        if any(root.has(sp.I) for root in roots):
            raise ValueError('Raices complejas (I) no soportadas para renderizado')

        unique_roots: List[sp.Expr] = []
        for root in roots:
            if root not in unique_roots:
                unique_roots.append(root)

        root_entries: List[Dict[str, str]] = []
        stability_entries: List[Dict[str, str]] = []
        existence_conditions: List[sp.Expr] = []
        has_always_real = False

        for root in unique_roots:
            root_str = Dynamic1DService._expr_to_str(root)
            deriv_at_root = sp.simplify(deriv_expr.subs(x, root))

            conditions = Dynamic1DService._extract_real_conditions(root)
            if not conditions:
                root_exists = 'siempre'
                if root.is_real is True:
                    has_always_real = True
                else:
                    if not root.has(sp.I):
                        has_always_real = True
            else:
                if len(conditions) == 1:
                    existence_conditions.append(conditions[0])
                    root_exists = Dynamic1DService._expr_to_str(conditions[0])
                else:
                    merged = sp.And(*conditions)
                    existence_conditions.append(merged)
                    root_exists = Dynamic1DService._expr_to_str(merged)

            root_entries.append({
                'root': root_str,
                'existence': root_exists,
            })

            stable_cond = None
            unstable_cond = None
            try:
                stable_cond = sp.solve_univariate_inequality(deriv_at_root < 0, p, relational=True)
            except Exception:
                stable_cond = None
            try:
                unstable_cond = sp.solve_univariate_inequality(deriv_at_root > 0, p, relational=True)
            except Exception:
                unstable_cond = None

            stability_entries.append({
                'root': root_str,
                'derivative': Dynamic1DService._expr_to_str(deriv_at_root),
                'stable_when': Dynamic1DService._format_inequality_result(stable_cond),
                'unstable_when': Dynamic1DService._format_inequality_result(unstable_cond),
            })

        existence_expr = None
        if has_always_real:
            existence_expr = 'siempre'
        elif existence_conditions:
            existence_expr = Dynamic1DService._expr_to_str(sp.simplify(sp.Or(*existence_conditions)))

        return {
            'existence': existence_expr,
            'roots': root_entries,
            'stability': stability_entries,
            'derivative': Dynamic1DService._expr_to_str(deriv_expr),
        }

    @staticmethod
    def _build_expr(model: str, func_str: str, params: Dict[str, float], control_enabled: bool) -> Tuple[str, Dict[str, float]]:
        if model == 'malthus':
            base = 'r*x'
        elif model == 'verhulst':
            base = 'mu*x*(1 - x/K)'
        elif model == 'newton':
            base = '-k*(x - Ta)'
        else:
            base = func_str

        if control_enabled and model in {'verhulst', 'custom'}:
            base = f'({base}) - h'

        return base, params

    @staticmethod
    def _compile_function(expr_str: str, params: Dict[str, float], allowed_symbols: List[str] = None) -> Tuple[Callable, str]:
        allowed = allowed_symbols or ['x']
        if params:
            allowed = list(dict.fromkeys(allowed + list(params.keys())))
        expr = Dynamic1DService._parse_expression(expr_str, params, allowed)

        if params:
            subs_map = {sp.Symbol(key, real=True): value for key, value in params.items()}
            expr = expr.subs(subs_map)

        free_symbols = {str(sym) for sym in expr.free_symbols}
        if free_symbols and free_symbols != {'x'}:
            missing = ', '.join(sorted(free_symbols - {'x'}))
            raise ValueError(f"Faltan parametros: {missing}")

        x = sp.Symbol('x')
        func = sp.lambdify(x, expr, 'numpy')
        return func, str(expr)

    @staticmethod
    def _safe_eval(f: Callable, x: np.ndarray) -> np.ndarray:
        with np.errstate(all='ignore'):
            y = f(x)
        y = np.asarray(y, dtype=float)
        if y.ndim == 0:
            # Función constante (p. ej. f(x) ≡ 0): el lambdify devuelve escalar.
            y = np.full(np.shape(x), float(y), dtype=float)
        y = y.copy()
        y[~np.isfinite(y)] = np.nan
        return y

    @staticmethod
    def _unique_sorted(values: List[float], tol: float = 1e-5) -> List[float]:
        if not values:
            return []
        values_sorted = sorted(values)
        unique_vals = [values_sorted[0]]
        for val in values_sorted[1:]:
            if abs(val - unique_vals[-1]) > tol:
                unique_vals.append(val)
        return unique_vals

    @staticmethod
    def find_equilibria(f: Callable, x_min: float, x_max: float, n: int = 400, tol: float = 1e-6) -> List[float]:
        xs = np.linspace(x_min, x_max, n)
        ys = Dynamic1DService._safe_eval(f, xs)

        roots: List[float] = []
        for i in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            if np.isnan(y0) or np.isnan(y1):
                continue

            if abs(y0) < tol:
                roots.append(float(xs[i]))
            if y0 * y1 < 0:
                try:
                    root = brentq(lambda z: float(f(z)), xs[i], xs[i + 1])
                    roots.append(float(root))
                except Exception:
                    continue
            if abs(y1) < tol:
                roots.append(float(xs[i + 1]))

        return Dynamic1DService._unique_sorted(roots, tol=1e-4)

    @staticmethod
    def classify_equilibria(f: Callable, roots: List[float]) -> List[Dict[str, Any]]:
        results = []
        df_tol = 1e-4
        f_tol = 1e-6
        for root in roots:
            try:
                df = float(numerical_derivative(f, root, order=1, h=1e-5))
            except Exception:
                df = float('nan')

            stability = 'indeterminado'
            reason = 'sin datos suficientes'
            if math.isfinite(df) and abs(df) > df_tol:
                if df < 0:
                    stability = 'estable'
                    reason = f"f'(x*) = {df:.6g} < 0"
                else:
                    stability = 'inestable'
                    reason = f"f'(x*) = {df:.6g} > 0"
            else:
                delta = max(1e-4, abs(root) * 1e-3)
                try:
                    left = float(f(root - delta))
                    right = float(f(root + delta))
                    if abs(left) < f_tol:
                        left = 0.0
                    if abs(right) < f_tol:
                        right = 0.0
                    if left > 0 and right < 0:
                        stability = 'estable'
                        reason = f"f(x*-d)>0 y f(x*+d)<0, d={delta:g}"
                    elif left < 0 and right > 0:
                        stability = 'inestable'
                        reason = f"f(x*-d)<0 y f(x*+d)>0, d={delta:g}"
                    elif left == 0.0 and right == 0.0:
                        stability = 'indeterminado'
                        reason = f"f(x*±d)≈0, d={delta:g}"
                    elif math.isfinite(left) and math.isfinite(right):
                        stability = 'semiestable'
                        reason = f"f(x*-d) y f(x*+d) no cambian de signo, d={delta:g}"
                except Exception:
                    stability = 'indeterminado'

            results.append({
                'x': root,
                'fprime': df if math.isfinite(df) else None,
                'stability': stability,
                'stability_reason': reason,
            })

        return results

    @staticmethod
    def phase_data(f: Callable, x_min: float, x_max: float, n: int = 400) -> Dict[str, Any]:
        xs = np.linspace(x_min, x_max, n)
        ys = Dynamic1DService._safe_eval(f, xs)

        flow_points: List[Dict[str, float]] = []
        flow_xs = np.linspace(x_min, x_max, 14)
        flow_vals = Dynamic1DService._safe_eval(f, flow_xs)
        for x_val, f_val in zip(flow_xs, flow_vals):
            if np.isnan(f_val):
                continue
            direction = 1 if f_val > 0 else -1 if f_val < 0 else 0
            flow_points.append({'x': float(x_val), 'dir': direction})

        return {
            'x': xs.tolist(),
            'fx': ys.tolist(),
            'flow': flow_points,
        }

    @staticmethod
    def time_solutions(f: Callable, initials: List[float], t_max: float, n: int = 200) -> Dict[str, Any]:
        t_eval = np.linspace(0, t_max, n)
        series = []
        for x0 in initials:
            sol = solve_ivp(
                lambda t, x: f(x),
                (0, t_max),
                [x0],
                t_eval=t_eval,
                method='RK45',
            )
            series.append({
                'x0': float(x0),
                'x': sol.y[0].tolist() if sol.success else [],
            })

        return {
            't': t_eval.tolist(),
            'series': series,
        }

    @staticmethod
    def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
        model = payload.get('model', 'custom')
        func_str = payload.get('func_str', 'x')
        params = payload.get('params', {}) or {}
        control_enabled = bool(payload.get('control_enabled', False))

        expr_str, params = Dynamic1DService._build_expr(model, func_str, params, control_enabled)
        f, expr_compiled = Dynamic1DService._compile_function(expr_str, params)

        x_min = float(payload.get('x_min', -1))
        x_max = float(payload.get('x_max', 3))
        t_max = float(payload.get('t_max', 10))
        n_phase = int(payload.get('n_phase', 400))
        n_time = int(payload.get('n_time', 200))
        initials = payload.get('initial_conditions', [0.5])

        roots = Dynamic1DService.find_equilibria(f, x_min, x_max, n=n_phase)
        equilibria = Dynamic1DService.classify_equilibria(f, roots)
        phase = Dynamic1DService.phase_data(f, x_min, x_max, n=n_phase)
        time = Dynamic1DService.time_solutions(f, initials, t_max, n=n_time)

        return {
            'model': model,
            'equation': expr_compiled,
            'params': params,
            'equilibria': equilibria,
            'phase': phase,
            'time': time,
        }

    @staticmethod
    def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
        model = payload.get('model', 'custom')
        func_str = payload.get('func_str', 'x')
        params = payload.get('params', {}) or {}
        control_enabled = bool(payload.get('control_enabled', False))

        expr_str, params = Dynamic1DService._build_expr(model, func_str, params, control_enabled)
        f, expr_compiled = Dynamic1DService._compile_function(expr_str, params)
        _ = f(0.0)

        return {
            'ok': True,
            'equation': expr_compiled,
            'model': model,
        }

    @staticmethod
    def bifurcation(payload: Dict[str, Any]) -> Dict[str, Any]:
        model = payload.get('model', 'custom')
        func_str = payload.get('func_str', 'x')
        params = payload.get('params', {}) or {}
        control_enabled = bool(payload.get('control_enabled', False))

        model_hint = payload.get('bif_model', model)

        raw_bif_param = payload.get('bif_param', 'r')
        bif_param = (raw_bif_param or 'r').strip() or 'r'
        x_min = float(payload.get('x_min', -1))
        x_max = float(payload.get('x_max', 3))
        n_phase = int(payload.get('n_phase', 400))

        bif_min = float(payload.get('bif_min', -1))
        bif_max = float(payload.get('bif_max', 1))
        bif_steps = int(payload.get('bif_steps', 60))

        phase_params = payload.get('phase_params', []) or []
        if not phase_params:
            # Si no se proporcionan, calcular puntos críticos automáticos
            center = (bif_min + bif_max) / 2
            delta = (bif_max - bif_min) / 4
            phase_params = [center - delta, center, center + delta]

        expr_str, _ = Dynamic1DService._build_expr(model, func_str, params, control_enabled)
        bif_allowed = list(dict.fromkeys(['x', bif_param] + list(params.keys())))
        expr_template = Dynamic1DService._parse_expression(expr_str, params, bif_allowed)
        if params:
            expr_template = expr_template.subs(params)

        exact_analysis = None
        try:
            exact_analysis = Dynamic1DService._exact_bifurcation_analysis(expr_template, bif_param)
            exact_analysis['model_hint'] = model_hint
        except Exception:
            exact_analysis = None

        param_values = np.linspace(bif_min, bif_max, bif_steps)
        equilibria_rows: List[Dict[str, Any]] = []

        # Intentar Inversión de Dominio (Domain Inversion)
        inverse_success = False
        try:
            x_sym = sp.Symbol('x', real=True)
            p_sym = sp.Symbol(bif_param, real=True)
            r_exprs = sp.solve(expr_template, p_sym)
            
            if len(r_exprs) == 1:
                r_expr = r_exprs[0]
                r_func = sp.lambdify(x_sym, r_expr, 'numpy')
                
                df_dx_expr = sp.diff(expr_template, x_sym)
                df_dx_func = sp.lambdify((x_sym, p_sym), df_dx_expr, 'numpy')
                
                # Grilla densa en el espacio de estados
                x_grid = np.linspace(x_min, x_max, n_phase * 5)
                
                for x_val in x_grid:
                    x_val_float = float(x_val)
                    try:
                        r_val = float(r_func(x_val_float))
                    except Exception:
                        continue
                        
                    if not math.isfinite(r_val):
                        continue
                        
                    if bif_min <= r_val <= bif_max:
                        try:
                            df = float(df_dx_func(x_val_float, r_val))
                        except Exception:
                            df = float('nan')
                            
                        stability = 'indeterminado'
                        reason = 'sin datos suficientes'
                        if math.isfinite(df) and abs(df) > 1e-4:
                            if df < 0:
                                stability = 'estable'
                                reason = f"f'(x*) = {df:.6g} < 0"
                            else:
                                stability = 'inestable'
                                reason = f"f'(x*) = {df:.6g} > 0"
                                
                        equilibria_rows.append({
                            'param': r_val,
                            'x': x_val_float,
                            'fprime': df if math.isfinite(df) else None,
                            'stability': stability,
                            'stability_reason': reason,
                        })
                
                if equilibria_rows:
                    inverse_success = True
                    # Ordenar por el parametro para facilitar el graficado
                    equilibria_rows.sort(key=lambda item: item['param'])
        except Exception:
            pass

        if not inverse_success:
            for p_val in param_values:
                local_params = dict(params)
                local_params[bif_param] = float(p_val)
                f, _ = Dynamic1DService._compile_function(expr_str, local_params)
                roots = Dynamic1DService.find_equilibria(f, x_min, x_max, n=n_phase)
                equilibria = Dynamic1DService.classify_equilibria(f, roots)
                for eq in equilibria:
                    equilibria_rows.append({
                        'param': float(p_val),
                        'x': eq['x'],
                        'fprime': eq['fprime'],
                        'stability': eq['stability'],
                        'stability_reason': eq.get('stability_reason'),
                    })

        phase_slices: List[Dict[str, Any]] = []
        for p_val in phase_params:
            local_params = dict(params)
            local_params[bif_param] = float(p_val)
            f, _ = Dynamic1DService._compile_function(expr_str, local_params)
            roots = Dynamic1DService.find_equilibria(f, x_min, x_max, n=n_phase)
            equilibria = Dynamic1DService.classify_equilibria(f, roots)
            phase = Dynamic1DService.phase_data(f, x_min, x_max, n=n_phase)
            phase_slices.append({
                'param': float(p_val),
                'equilibria': equilibria,
                'phase': phase,
            })

        return {
            'model': model,
            'equation': str(expr_template),
            'bif_param': bif_param,
            'bifurcation': {
                'param_values': param_values.tolist(),
                'equilibria': equilibria_rows,
            },
            'phase_slices': phase_slices,
            'exact_analysis': exact_analysis,
        }
