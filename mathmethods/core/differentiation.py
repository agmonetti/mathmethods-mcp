"""
Derivación Numérica - Diferencias Finitas
- Primera derivada
- Segunda derivada
"""
import sympy as sp
import warnings
from typing import Dict
from typing import Dict, Callable
from .utils import safe_sympify

class DifferentiationService:
    
    # --- TUS MÉTODOS ORIGINALES (INTACTOS POR COMPATIBILIDAD) ---
    @staticmethod
    def compilar_funcion(texto_funcion: str) -> Callable:
        """Convierte string a función callable."""
        try:
            x = sp.Symbol('x')
            expr = safe_sympify(texto_funcion, allowed_symbols=['x'])
            return sp.lambdify(x, expr, 'numpy'), expr
        except Exception as e:
            raise ValueError(f"Error compilando función: {str(e)}")
    
    @staticmethod
    def diferencias_finitas(f: Callable, x_val: float, h: float = 1e-5, precision: int = 8) -> Dict:
        """Calcula primera y segunda derivada con diferencias centrales."""
        f1_aprox = (f(x_val + h) - f(x_val - h)) / (2 * h)
        f2_aprox = (f(x_val + h) - 2*f(x_val) + f(x_val - h)) / (h**2)
        return {
            "x": round(x_val, precision), "h": round(h, precision),
            "primera_derivada": {"aproximada": round(f1_aprox, precision), "formula": "(f(x+h) - f(x-h)) / (2h)"},
            "segunda_derivada": {"aproximada": round(f2_aprox, precision), "formula": "(f(x+h) - 2f(x) + f(x-h)) / h²"}
        }
    
    @staticmethod
    def diferencias_finitas_con_exacto(f_str: str, x_val: float, h: float = 1e-5, precision: int = 8) -> Dict:
        """Calcula derivadas numéricas Y las compara con exactas."""
        f, expr = DifferentiationService.compilar_funcion(f_str)
        x = sp.Symbol('x')
        df1_expr = sp.diff(expr, x)
        df2_expr = sp.diff(df1_expr, x)
        f1_exacta = float(df1_expr.subs(x, x_val).evalf())
        f2_exacta = float(df2_expr.subs(x, x_val).evalf())
        f1_aprox = (f(x_val + h) - f(x_val - h)) / (2 * h)
        f2_aprox = (f(x_val + h) - 2*f(x_val) + f(x_val - h)) / (h**2)
        # Aquí también deberíamos usar .evalf() si queremos máxima precisión, pero el problema
        # estaba en el pre-redondeo que hacíamos para la nueva web.
        error_1 = abs(f1_exacta - f1_aprox)
        error_2 = abs(f2_exacta - f2_aprox)
        return {
            "x": round(x_val, precision), "h": round(h, precision),
            "primera_derivada": {"numerica": round(f1_aprox, precision), "exacta": round(f1_exacta, precision), "error": round(error_1, 8), "formula_numerica": "(f(x+h) - f(x-h)) / (2h)"},
            "segunda_derivada": {"numerica": round(f2_aprox, precision), "exacta": round(f2_exacta, precision), "error": round(error_2, 8), "formula_numerica": "(f(x+h) - 2f(x) + f(x-h)) / h²"}
        }

    # --- NUESTRO MÉTODO NUEVO UNIFICADO (ACTUALIZADO PARA FORMATOS) ---
    @staticmethod
    def calcular_diferencias_completas(func_str: str, x_val: float, h: float, precision: int = 8) -> Dict:
        """Calcula diferencias Progresivas, Regresivas y Centrales enviando error crudo."""
        try:
            # Tratamiento especial para 'log' en SymPy (por defecto es ln)
            # si el usuario escribe 'log10', SymPy lo manejará. 'ln' es el estándar.
            
            x_sym = sp.Symbol('x')
            diccionario_local = {'e': sp.E, 'pi': sp.pi}
            expr = safe_sympify(func_str, local_dict=diccionario_local, allowed_symbols=['x'])
            f = sp.lambdify(x_sym, expr, 'numpy')
            
            dev_expr = sp.diff(expr, x_sym)
            dev_exacta = float(dev_expr.subs(x_sym, x_val).evalf()) 
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fx = float(f(x_val))
                fx_mas_h = float(f(x_val + h))
                fx_menos_h = float(f(x_val - h))

            progresiva = (fx_mas_h - fx) / h
            regresiva = (fx - fx_menos_h) / h
            central = (fx_mas_h - fx_menos_h) / (2 * h)
            
            dev2_expr = sp.diff(dev_expr, x_sym)
            dev2_exacta = float(dev2_expr.subs(x_sym, x_val).evalf())
            segunda_central = (fx_mas_h - 2*fx + fx_menos_h) / (h**2)

            return {
                "x_evaluado": round(x_val, precision),
                "h_usado": h,
                "f_x": round(fx, precision),
                "derivada_exacta": round(dev_exacta, precision),
                "segunda_derivada_exacta": round(dev2_exacta, precision),
                "resultados": [
                    {
                        "metodo": "Diferencia Progresiva",
                        "formula": "[f(x+h) - f(x)] / h",
                        "valor": round(progresiva, precision),
                        # CONEXIÓN: Enviamos el error crudo sin redondear
                        "error_raw": abs(dev_exacta - progresiva)
                    },
                    {
                        "metodo": "Diferencia Regresiva",
                        "formula": "[f(x) - f(x-h)] / h",
                        "valor": round(regresiva, precision),
                        "error_raw": abs(dev_exacta - regresiva)
                    },
                    {
                        "metodo": "Diferencia Central",
                        "formula": "[f(x+h) - f(x-h)] / 2h",
                        "valor": round(central, precision),
                        "error_raw": abs(dev_exacta - central)
                    },
                    {
                        "metodo": "Segunda Derivada (Central)",
                        "formula": "[f(x+h) - 2f(x) + f(x-h)] / h²",
                        "valor": round(segunda_central, precision),
                        "error_raw": abs(dev2_exacta - segunda_central)
                    }
                ]
            }
        except Exception as e:
            raise ValueError(f"Error evaluando la función: {str(e)}")