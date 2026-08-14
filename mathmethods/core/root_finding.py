"""
Métodos para búsqueda de raíces
- Bisección
- Punto Fijo
- Newton-Raphson
- Aitken (Aceleración)
"""
import numpy as np
from typing import Dict, List, Tuple, Callable
import sympy as sp
import math
import re

class RootFindingService:
    
    @staticmethod
    def compilar_funcion(texto_funcion: str, variables: str = 'x') -> Callable:
        """Convierte string matemático a función callable con NumPy."""
        try:
            texto_funcion = re.sub(r'(?<![A-Za-z0-9_])[eE]\^', 'E**', texto_funcion).replace('^', '**')
            x = sp.Symbol(variables.split()[0])
            diccionario_local = {'e': sp.E, 'pi': sp.pi}
            expr = sp.sympify(texto_funcion, locals=diccionario_local)
            return sp.lambdify(x, expr, 'numpy')
        except Exception as e:
            raise ValueError(f"Error compilando función: {str(e)}")
    
    @staticmethod
    def biseccion(f: Callable, a: float, b: float, 
                  tol: float = 1e-6, max_iter: int = 100, 
                  precision: int = 8) -> Dict:
        """Método de Bisección - acecha cambios de signo."""
        try:
            fa = float(f(a))
            fb = float(f(b))
        except:
            raise ValueError("Error al evaluar f(a) o f(b). Asegúrese de que la función es válida en el intervalo.")
            
        if fa * fb >= 0:
            raise ValueError("f(a) y f(b) deben tener signos opuestos (f(a) * f(b) < 0).")
        
        iteraciones = []
        
        for i in range(max_iter):
            c = (a + b) / 2.0
            try:
                fc = float(f(c))
                if math.isnan(fc) or math.isinf(fc):
                    raise OverflowError("El valor de f(c) divergió a infinito o no existe.")
            except Exception as e:
                return {
                    "metodo": "Bisección", "raiz": None, "iteraciones": iteraciones,
                    "convergencia": False, "num_iter": i, "error_msg": f"Divergencia matemática: {str(e)}"
                }
            
            # ELIMINAMOS TODOS LOS REDONDEOS ACÁ (Enviamos datos crudos)
            error_calc = abs(b - a) / 2.0
            iteraciones.append({
                "i": i + 1,
                "a": a,
                "b": b,
                "c": c,
                "f_c": fc,
                "error": error_calc
            })
            
            if abs(fc) < tol or error_calc < tol:
                return {
                    "metodo": "Bisección",
                    "raiz": round(c, precision), # La respuesta final sí se puede redondear a gusto
                    "iteraciones": iteraciones,
                    "convergencia": True,
                    "num_iter": i + 1
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return {
            "metodo": "Bisección",
            "raiz": round((a + b) / 2.0, precision),
            "iteraciones": iteraciones,
            "convergencia": False,
            "num_iter": max_iter,
            "error_msg": "Se alcanzó el límite de iteraciones sin converger."
        }
    
    @staticmethod
    def punto_fijo(g: Callable, x0: float, 
                   tol: float = 1e-6, max_iter: int = 100,
                   precision: int = 8) -> Dict:
        """Método de Punto Fijo - x = g(x)."""
        x = x0
        iteraciones = []
        verificacion = RootFindingService._verificar_lipschitz(g, x0)
        
        for i in range(max_iter):
            try:
                x_new = float(g(x))
                if math.isnan(x_new) or math.isinf(x_new):
                    raise OverflowError("La iteración explotó hacia el infinito.")
            except Exception as e:
                return {
                    "metodo": "Punto Fijo",
                    "raiz": None,
                    "iteraciones": iteraciones,
                    "verificacion": verificacion,
                    "convergencia": False,
                    "num_iter": i,
                    "error_msg": f"¡La función diverge! Asegúrate de que |g'(x)| < 1. ({str(e)})"
                }

            error = abs(x_new - x)
            
            # ELIMINAMOS TODOS LOS REDONDEOS ACÁ
            iteraciones.append({
                "i": i + 1,
                "x": x,
                "g_x": x_new,
                "error": error
            })
            
            if error < tol:
                return {
                    "metodo": "Punto Fijo",
                    "raiz": round(x_new, precision),
                    "iteraciones": iteraciones,
                    "verificacion": verificacion,
                    "convergencia": True,
                    "num_iter": i + 1
                }
            
            x = x_new
        
        return {
            "metodo": "Punto Fijo",
            "raiz": round(x, precision),
            "iteraciones": iteraciones,
            "verificacion": verificacion,
            "convergencia": False,
            "num_iter": max_iter,
            "error_msg": "No converge. Revisa si |g'(x)| < 1 en el intervalo."
        }
    
    @staticmethod
    def _derivada_numerica(f: Callable, x: float, dx: float = 1e-6) -> float:
        """Aproxima derivada con diferencias centrales."""
        return (f(x + dx) - f(x - dx)) / (2.0 * dx)

    @staticmethod
    def _verificar_lipschitz(g: Callable, x0: float, dx: float = 1e-6) -> Dict:
        """Verifica condición local de contracción en la semilla: |g'(x0)| < 1."""
        g_prime = float(RootFindingService._derivada_numerica(g, x0, dx))
        return {
            "x0": x0,
            "g_prime": g_prime,
            "abs_g_prime": abs(g_prime),
            "cumple": abs(g_prime) < 1,
        }
    
    @staticmethod
    def newton_raphson(f: Callable, x0: float,
                       tol: float = 1e-6, max_iter: int = 100,
                       precision: int = 8) -> Dict:
        """Método de Newton-Raphson - usa derivada numérica."""
        x = x0
        iteraciones = []
        
        for i in range(max_iter):
            try:
                fx = float(f(x))
                dfx = float(RootFindingService._derivada_numerica(f, x))
                
                if abs(dfx) < 1e-12:
                    raise ValueError("La derivada se hizo cero (tangente horizontal).")
                
                x_new = x - fx / dfx
                if math.isnan(x_new) or math.isinf(x_new):
                    raise OverflowError("La iteración divergió a infinito.")
            except Exception as e:
                return {
                    "metodo": "Newton-Raphson",
                    "raiz": None,
                    "iteraciones": iteraciones,
                    "convergencia": False,
                    "num_iter": i,
                    "error_msg": f"Falla en la iteración: {str(e)}"
                }
                
            error = abs(x_new - x)
            
            # ELIMINAMOS TODOS LOS REDONDEOS ACÁ
            iteraciones.append({
                "i": i + 1,
                "x": x,
                "f (x)": fx,
                "f ' (x)": dfx,
                "Resultado": x_new,
                "error": error
            })
            
            if error < tol:
                return {
                    "metodo": "Newton-Raphson",
                    "raiz": round(x_new, precision),
                    "iteraciones": iteraciones,
                    "convergencia": True,
                    "num_iter": i + 1
                }
            
            x = x_new
        
        return {
            "metodo": "Newton-Raphson",
            "raiz": round(x, precision),
            "iteraciones": iteraciones,
            "convergencia": False,
            "num_iter": max_iter,
            "error_msg": "Se alcanzó el límite de iteraciones sin converger."
        }
    
    @staticmethod
    def aitken(g: Callable, x0: float,
               tol: float = 1e-6, max_iter: int = 100,
               precision: int = 8) -> Dict:
        """Aceleración de Aitken sobre punto fijo."""
        x = x0
        iteraciones = []
        verificacion = RootFindingService._verificar_lipschitz(g, x0)
        
        for i in range(max_iter):
            try:
                x1 = float(g(x))
                x2 = float(g(x1))
                
                denominador = x2 - 2*x1 + x
                if abs(denominador) > 1e-12:
                    x_acelerado = x - (x1 - x)**2 / denominador
                else:
                    x_acelerado = x2
                    
                if math.isnan(x_acelerado) or math.isinf(x_acelerado):
                    raise OverflowError("La aceleración divergió a infinito.")
            except Exception as e:
                return {
                    "metodo": "Aitken",
                    "raiz": None,
                    "iteraciones": iteraciones,
                    "verificacion": verificacion,
                    "convergencia": False,
                    "num_iter": i,
                    "error_msg": f"¡Divergencia detectada! {str(e)}"
                }
            
            error = abs(x_acelerado - x)
            
            # ELIMINAMOS TODOS LOS REDONDEOS ACÁ
            iteraciones.append({
                "i": i + 1,
                "x": x,
                "g_x": x1,
                "g_g_x": x2,
                "x_acelerado": x_acelerado,
                "error": error
            })
            
            if error < tol:
                return {
                    "metodo": "Aitken",
                    "raiz": round(x_acelerado, precision),
                    "iteraciones": iteraciones,
                    "verificacion": verificacion,
                    "convergencia": True,
                    "num_iter": i + 1
                }
            
            x = x_acelerado
        
        return {
            "metodo": "Aitken",
            "raiz": round(x, precision),
            "iteraciones": iteraciones,
            "verificacion": verificacion,
            "convergencia": False,
            "num_iter": max_iter,
            "error_msg": "No converge."
        }
    @staticmethod
    def comparar_metodos(func_str: str, g_str: str, a: float, b: float, x0: float, 
                         tol: float = 1e-6, max_iter: int = 100, precision: int = 8) -> Dict:
        """
        Orquestador que ejecuta los 4 métodos en paralelo y devuelve un resumen comparativo.
        """
        resultados = {
            "biseccion": None,
            "punto_fijo": None,
            "newton_raphson": None,
            "aitken": None
        }

        # 1. Bisección
        try:
            f_bisec = RootFindingService.compilar_funcion(func_str)
            resultados["biseccion"] = RootFindingService.biseccion(f_bisec, a, b, tol, max_iter, precision)
        except Exception as e:
            resultados["biseccion"] = {"convergencia": False, "error_msg": str(e), "iteraciones": []}

        # 2. Punto Fijo
        try:
            g_pf = RootFindingService.compilar_funcion(g_str)
            resultados["punto_fijo"] = RootFindingService.punto_fijo(g_pf, x0, tol, max_iter, precision)
        except Exception as e:
            resultados["punto_fijo"] = {"convergencia": False, "error_msg": str(e), "iteraciones": []}

        # 3. Newton-Raphson
        try:
            f_nr = RootFindingService.compilar_funcion(func_str)
            resultados["newton_raphson"] = RootFindingService.newton_raphson(f_nr, x0, tol, max_iter, precision)
        except Exception as e:
            resultados["newton_raphson"] = {"convergencia": False, "error_msg": str(e), "iteraciones": []}

        # 4. Aitken
        try:
            g_aitken = RootFindingService.compilar_funcion(g_str)
            resultados["aitken"] = RootFindingService.aitken(g_aitken, x0, tol, max_iter, precision)
        except Exception as e:
            resultados["aitken"] = {"convergencia": False, "error_msg": str(e), "iteraciones": []}

        return {
            "parametros": {
                "f(x)": func_str, "g(x)": g_str, "a": a, "b": b, "x0": x0, "tol": tol
            },
            "comparativa": resultados
        }