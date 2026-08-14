import numpy as np
import sympy as sp
import math
import re

class ODEService:

    @staticmethod
    def _round_numeric(value, precision):
        if isinstance(value, dict):
            return {k: ODEService._round_numeric(v, precision) for k, v in value.items()}
        if isinstance(value, list):
            return [ODEService._round_numeric(v, precision) for v in value]
        if isinstance(value, (float, np.floating)):
            return round(float(value), precision)
        return value
    
    @staticmethod
    def resolver_edo_exacta(texto_dy_dx, x0_val, y0_val):
        """Usa SymPy para resolver la EDO analíticamente y retorna un evaluador seguro."""
        x = sp.Symbol('x')
        y_func = sp.Function('y')(x)
        
        diccionario_local = {'y': y_func, 'e': sp.E, 'pi': sp.pi}
        texto_dy_dx = re.sub(r'(?<![A-Za-z0-9_])[eE]\^', 'E**', texto_dy_dx).replace('^', '**')
        f_expr = sp.sympify(texto_dy_dx, locals=diccionario_local)
        
        edo = sp.Eq(y_func.diff(x), f_expr)
        condiciones_iniciales = {y_func.subs(x, x0_val): y0_val}
        
        solucion_general = sp.dsolve(edo, ics=condiciones_iniciales)
        expr_exacta = solucion_general.rhs
        
        f_exacta = sp.lambdify(x, expr_exacta, 'numpy')
        
        def f_exacta_segura(x_val):
            return float(f_exacta(x_val))
            
        return f_exacta_segura, str(expr_exacta)

    @staticmethod
    def compilar_f(ecuacion_str):
        """Compila f(x,y) a función lambda de NumPy."""
        x_sym, y_sym = sp.symbols('x y')
        ecuacion_str_limpia = re.sub(r'(?<![A-Za-z0-9_])[eE]\^', 'E**', ecuacion_str).replace('^', '**')
        return sp.lambdify((x_sym, y_sym), sp.sympify(ecuacion_str_limpia, locals={'e': sp.E, 'pi': sp.pi}), 'numpy')

    @staticmethod
    def ejecutar_metodo(metodo, ecuacion_str, x0, y0, xf, h, precision=8, tol=None):
        """
        Orquestador principal. Ejecuta el método solicitado, 
        calcula la solución exacta, los errores y empaqueta todo para la API.
        """
        f = ODEService.compilar_f(ecuacion_str)

        f_exacta = None
        expr_exacta_str = None
        try:
            f_exacta, expr_exacta_str = ODEService.resolver_edo_exacta(ecuacion_str, x0, y0)
        except Exception:
            f_exacta = None
            expr_exacta_str = None

        if h <= 0:
            raise ValueError("h debe ser positivo.")

        n_steps = int(round((xf - x0) / h))
        x_values = np.linspace(x0, xf, n_steps + 1)
        x_values = np.round(x_values, decimals=10)
        n_steps = len(x_values)

        if f_exacta is not None:
            y_exacta = np.array([f_exacta(xi) for xi in x_values])
        else:
            y_exacta = np.full(n_steps, np.nan)
        
        resultado = {
            "ecuacion": ecuacion_str,
            "solucion_exacta_str": expr_exacta_str,
            "h": h,
            "tol": tol,
            "tabla": [],
            "x_plot": x_values.tolist(),
            "y_exacta_plot": y_exacta.tolist() if f_exacta is not None else []
        }

        # Arrays para los cálculos
        y_n = np.zeros(n_steps)
        y_n[0] = y0

        if metodo == "euler":
            y_n1 = np.zeros(n_steps)
            for i in range(n_steps):
                if i > 0: y_n[i] = y_n1[i-1]
                y_n1[i] = y_n[i] + h * f(x_values[i], y_n[i])
                
                err = abs(y_exacta[i] - y_n[i]) if f_exacta is not None else None
                resultado["tabla"].append({
                    "i": i, "xn": x_values[i], "yn": y_n[i], 
                    "yn1": y_n1[i], # <-- ACÁ ESTÁ EL ARREGLO: Ya no devolvemos None al final
                    "yr": y_exacta[i] if f_exacta is not None else None, "error": err if i > 0 else 0
                })
            resultado["y_plot"] = y_n.tolist()

        elif metodo == "heun":
            y_pred = np.zeros(n_steps)
            y_corr = np.zeros(n_steps)
            for i in range(n_steps):
                if i > 0: y_n[i] = y_corr[i-1]
                
                if i < n_steps - 1:
                    f_xy = f(x_values[i], y_n[i])
                    y_pred[i] = y_n[i] + h * f_xy
                    y_corr[i] = y_n[i] + (h / 2.0) * (f_xy + f(x_values[i+1], y_pred[i]))
                
                err = abs(y_exacta[i] - y_n[i]) if f_exacta is not None else None
                resultado["tabla"].append({
                    "i": i, "xn": x_values[i], "yn": y_n[i],
                    "y_pred": y_pred[i] if i < n_steps - 1 else None,
                    "y_corr": y_corr[i] if i < n_steps - 1 else None,
                    "yr": y_exacta[i] if f_exacta is not None else None, "error": err if i > 0 else 0
                })
            resultado["y_plot"] = y_n.tolist()

        elif metodo == "rk4":
            k1 = np.zeros(n_steps); k2 = np.zeros(n_steps)
            k3 = np.zeros(n_steps); k4 = np.zeros(n_steps)
            y_n1 = np.zeros(n_steps)
            
            for i in range(n_steps):
                if i > 0: y_n[i] = y_n1[i-1]
                
                if i < n_steps - 1:
                    xn = x_values[i]; yn = y_n[i]
                    k1[i] = f(xn, yn)
                    k2[i] = f(xn + h/2.0, yn + (h/2.0) * k1[i])
                    k3[i] = f(xn + h/2.0, yn + (h/2.0) * k2[i])
                    k4[i] = f(xn + h, yn + h * k3[i])
                    y_n1[i] = yn + (h / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                
                err = abs(y_exacta[i] - y_n[i]) if f_exacta is not None else None
                resultado["tabla"].append({
                    "i": i, "xn": x_values[i], "yn": y_n[i],
                    "k1": k1[i] if i < n_steps - 1 else None, "k2": k2[i] if i < n_steps - 1 else None,
                    "k3": k3[i] if i < n_steps - 1 else None, "k4": k4[i] if i < n_steps - 1 else None,
                    "yn1": y_n1[i] if i < n_steps - 1 else None,
                    "yr": y_exacta[i] if f_exacta is not None else None, "error": err if i > 0 else 0
                })
            resultado["y_plot"] = y_n.tolist()

        final_error = resultado["tabla"][-1]["error"] if resultado["tabla"] else None
        resultado["acepta_tolerancia"] = bool(tol is not None and final_error is not None and final_error <= tol)

        precision = max(1, min(int(precision), 15))
        return ODEService._round_numeric(resultado, precision)