"""
Interpolación Numérica - Polinomio de Lagrange
"""
import numpy as np
from typing import Dict, List
import sympy as sp
import math
import re

class InterpolationService:
    
    @staticmethod
    def compilar_funcion(texto_funcion: str):
        """Convierte string a función simbólica."""
        try:
            texto_funcion = re.sub(r'(?<![A-Za-z0-9_])[eE]\^', 'E**', texto_funcion).replace('^', '**')
            x = sp.Symbol('x')
            diccionario_local = {'e': sp.E, 'pi': sp.pi}
            expr = sp.sympify(texto_funcion, locals=diccionario_local)
            return sp.lambdify(x, expr, 'numpy'), expr
        except Exception as e:
            raise ValueError(f"Error compilando: {str(e)}")
    
    @staticmethod
    def lagrange(puntos_x: List[float], x_eval: float = None,
                func_str: str = None, puntos_y: List[float] = None,
                precision: int = 8) -> Dict:
        
        x_sym = sp.Symbol('x')
        n = len(puntos_x)
        
        # Obtener los valores y
        if func_str:
            f, expr = InterpolationService.compilar_funcion(func_str)
            puntos_y = [float(f(xi)) for xi in puntos_x]
        elif puntos_y is None:
            raise ValueError("Proveer func_str o puntos_y")
        
        # Construir polinomio de Lagrange
        P = 0
        terminos_construccion = []
        
        for i in range(n):
            L_i = 1
            for j in range(n):
                if i != j:
                    L_i *= (x_sym - puntos_x[j]) / (puntos_x[i] - puntos_x[j])
            term = puntos_y[i] * L_i
            P += term
            
            # Guardamos el término para mostrarlo en el frontend
            terminos_construccion.append({
                "i": i,
                "l_i_y_i": str(sp.expand(term))
            })
        
        P_expanded = sp.expand(P)
        grado_sym = sp.degree(P_expanded, x_sym)
        # Parseo seguro del grado
        grado = int(grado_sym) if hasattr(grado_sym, 'is_number') and grado_sym.is_number and not getattr(grado_sym, 'is_infinite', False) else 0
        
        # Tabla de puntos
        tabla = []
        for xi, yi in zip(puntos_x, puntos_y):
            tabla.append({
                "x": round(xi, precision),
                "y": round(float(yi), precision)
            })
        
        resultado = {
            "metodo": "Lagrange",
            "puntos": tabla,
            "polinomio": str(P_expanded),
            "grado": grado,
            "terminos": terminos_construccion
        }
        
        # Si hay x_eval, evaluar
        if x_eval is not None:
            P_eval = float(P_expanded.subs(x_sym, x_eval).evalf())
            resultado["x_eval"] = x_eval
            resultado["P_eval"] = round(P_eval, precision)
            
            # Error local y cota global si tenemos función
            if func_str:
                f_eval = float(f(x_eval))
                error_local = abs(f_eval - P_eval)
                resultado["f_eval"] = round(f_eval, precision)
                resultado["error_local"] = round(error_local, 8)
                
                # --- CÁLCULO DE COTA DE ERROR GLOBAL ---
                derivada_orden = n
                derivada_expr = sp.diff(expr, x_sym, derivada_orden)
                
                # Rango para buscar máximos [min_x, max_x]
                intervalo_a = min(min(puntos_x), x_eval)
                intervalo_b = max(max(puntos_x), x_eval)
                x_vals = np.linspace(intervalo_a, intervalo_b, 1000)
                
                # Máximo de la derivada
                df_lambdify = sp.lambdify(x_sym, derivada_expr, 'numpy')
                df_eval = df_lambdify(x_vals)
                if np.isscalar(df_eval):
                    df_eval = np.full_like(x_vals, df_eval, dtype=float)
                max_derivada = float(np.max(np.abs(df_eval)))
                
                # Máximo de g(x) = prod(x - x_i)
                g_vals = np.ones_like(x_vals, dtype=float)
                for xi in puntos_x:
                    g_vals *= (x_vals - xi)
                max_g = float(np.max(np.abs(g_vals)))
                
                factorial_n = math.factorial(n)
                cota_global = (max_derivada / factorial_n) * max_g
                exito = cota_global >= error_local
                
                resultado["analisis_error"] = {
                    "derivada_orden": n,
                    "derivada_expr": str(derivada_expr).replace('**', '^'),
                    "max_derivada": max_derivada,
                    "max_g": max_g,
                    "factorial": factorial_n,
                    "cota_global": cota_global,
                    "exito": exito
                }
        
        return resultado