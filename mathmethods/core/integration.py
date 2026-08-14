"""
Integracion Numerica - Formulas de Newton-Cotes
- Trapecio (simple y compuesto)
- Rectangulo (punto medio compuesto)
- Simpson 1/3 (simple y compuesto)
- Simpson 3/8 (simple y compuesto)
"""
import numpy as np
import sympy as sp
from typing import Dict, Callable
import warnings
import re

class IntegrationService:
    
    @staticmethod
    def compilar_funcion(texto_funcion: str) -> Callable:
        try:
            texto_funcion = re.sub(r'(?<![A-Za-z0-9_])[eE]\^', 'E**', texto_funcion).replace('^', '**').replace('sen', 'sin')
            x_sym = sp.Symbol('x')
            diccionario_local = {'e': sp.E, 'pi': sp.pi}
            expr = sp.sympify(texto_funcion, locals=diccionario_local)
            func_base = sp.lambdify(x_sym, expr, 'numpy')
            
            def func_inteligente(x_val):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_val = func_base(x_val)
                
                # Intentar convertir a float, si genera complejos pasa a NaN
                try:
                    y_array = np.asarray(y_val, dtype=float)
                except TypeError:
                    y_array = np.full_like(x_val, np.nan, dtype=float)
                
                if np.isscalar(x_val) and y_array.ndim == 0:
                     if np.isnan(y_array) or np.isinf(y_array):
                         lim = sp.limit(expr, x_sym, x_val)
                         # Solo aceptamos limites reales finitos
                         if lim.is_finite and lim.is_real:
                             val = float(lim)
                             if getattr(func_inteligente, 'grabar_rescates', False):
                                 func_inteligente.limites_calculados.append((float(x_val), val))
                             return val
                         else:
                             raise ValueError(f"IMPROPIA: Indeterminación o resultado imaginario en x = {float(x_val):.4f}")
                     return float(y_array)
                     
                else:
                    if y_array.ndim == 0:
                        y_array = np.full_like(x_val, float(y_array))
                        
                    if np.any(np.isnan(y_array)) or np.any(np.isinf(y_array)):
                        intentos = 0
                        for i in range(len(y_array)):
                            if np.isnan(y_array[i]) or np.isinf(y_array[i]):
                                intentos += 1
                                if intentos > 15:
                                    raise ValueError("IMPROPIA: Función no definida o compleja en gran parte del intervalo.")
                                
                                lim = sp.limit(expr, x_sym, x_val[i])
                                if lim.is_finite and lim.is_real:
                                    val = float(lim)
                                    y_array[i] = val
                                    if getattr(func_inteligente, 'grabar_rescates', False):
                                        func_inteligente.limites_calculados.append((float(x_val[i]), val))
                                else:
                                    raise ValueError(f"IMPROPIA: Asíntota o raíz imaginaria en x = {x_val[i]:.4f}")
                    return y_array
            
            func_inteligente.limites_calculados = []
            func_inteligente.grabar_rescates = False
            func_inteligente.expr_sympy = expr
            func_inteligente.x_symbol = x_sym
            return func_inteligente
        except Exception as e:
            raise ValueError(f"Error compilando funcion: {str(e)}")

    @staticmethod
    def _orden_derivada_et(metodo: str) -> int:
        if metodo in ["rectangulo", "trapecio"]:
            return 2
        if metodo in ["simpson13", "simpson38"]:
            return 4
        raise ValueError("Método no soportado para error de truncamiento")

    @staticmethod
    def _coeficiente_et(a: float, b: float, n: int, metodo: str) -> float:
        if metodo == "rectangulo":
            return ((b - a) ** 3) / (24 * (n ** 2))
        if metodo == "trapecio":
            return -((b - a) ** 3) / (12 * (n ** 2))
        if metodo == "simpson13":
            return -((b - a) ** 5) / (180 * (n ** 4))
        if metodo == "simpson38":
            return -((b - a) ** 5) / (80 * (n ** 4))
        raise ValueError("Método no soportado para error de truncamiento")

    @staticmethod
    def _derivada_n_valor(expr: sp.Expr, x_sym: sp.Symbol, n: int, punto: float) -> float:
        return float(sp.diff(expr, x_sym, n).subs(x_sym, punto))

    @staticmethod
    def _calcular_error_truncamiento_epsilon(
        f: Callable,
        a: float,
        b: float,
        n: int,
        metodo: str,
        epsilon: float,
        precision: int,
    ):
        if epsilon is None:
            return "No definido (falta epsilon)"

        if epsilon < min(a, b) or epsilon > max(a, b):
            return "No definido (epsilon fuera del intervalo [a,b])"

        expr = getattr(f, "expr_sympy", None)
        x_sym = getattr(f, "x_symbol", None)
        if expr is None or x_sym is None:
            return "No definido (función simbólica no disponible)"

        try:
            orden = IntegrationService._orden_derivada_et(metodo)
            coeficiente = IntegrationService._coeficiente_et(a, b, n, metodo)
            derivada = IntegrationService._derivada_n_valor(expr, x_sym, orden, epsilon)
            et = coeficiente * derivada
            return {
                "epsilon": round(float(epsilon), precision),
                "orden_derivada": orden,
                "derivada_en_epsilon": round(float(derivada), precision),
                "error_truncamiento": round(float(et), precision),
            }
        except Exception:
            return "Indefinido matemáticamente en epsilon"

    @staticmethod
    def _verificar_asintota(y_array: np.ndarray):
        if np.any(np.isnan(y_array)) or np.any(np.isinf(y_array)):
            raise ValueError("IMPROPIA: Indeterminación o valores imaginarios detectados.")
        if np.any(np.abs(y_array) > 1e10):
            raise ValueError("IMPROPIA: Asíntota vertical detectada. El valor se dispara al infinito.")

    @staticmethod
    def _validar_integral(f: Callable, a: float, b: float):
        # Escaneo estrictamente interior para evitar colapsos con bordes complejos
        margen = (b - a) * 0.001
        x_test = np.linspace(a + margen, b - margen, 1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_test = f(x_test)
        IntegrationService._verificar_asintota(y_test)
    
    @staticmethod
    def _segunda_derivada_numerica(f: Callable, x: np.ndarray, dx: float = 1e-4) -> np.ndarray:
        return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx**2)
    
    @staticmethod
    def _cuarta_derivada_numerica(f: Callable, x: np.ndarray, dx: float = 1e-3) -> np.ndarray:
        return (f(x + 2*dx) - 4*f(x + dx) + 6*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (dx**4)
        
    @staticmethod
    def _calcular_cota_error(f: Callable, a: float, b: float, n: int, metodo: str, precision: int) -> str:
        """Calcula el error de forma super blindada evitando salir de los bordes."""
        try:
            # Apagamos los avisos de rescate para no ensuciar la libreta
            estado_previo = getattr(f, 'grabar_rescates', False)
            f.grabar_rescates = False 
            
            if metodo in ["rectangulo", "trapecio"]:
                dx = 1e-4
                margen = 2 * dx
                if b - a <= 2 * margen: return "Intervalo muy chico"
                x_fino = np.linspace(a + margen, b - margen, 1000)
                deriv = IntegrationService._segunda_derivada_numerica(f, x_fino, dx)
                deriv = deriv[~np.isnan(deriv)]
                if len(deriv) == 0: return "Indefinido"
                max_deriv = np.max(np.abs(deriv))
                
                if max_deriv > 1e7: return "Diverge (Derivada no acotada)"
                val = ((b - a)**3 / (24 * n**2)) if metodo == "rectangulo" else ((b - a)**3 / (12 * n**2))
                ans = round(float(val * max_deriv), precision)
                
            elif metodo in ["simpson13", "simpson38"]:
                dx = 1e-3
                margen = 3 * dx
                if b - a <= 2 * margen: return "Intervalo muy chico"
                x_fino = np.linspace(a + margen, b - margen, 1000)
                deriv = IntegrationService._cuarta_derivada_numerica(f, x_fino, dx)
                deriv = deriv[~np.isnan(deriv)]
                if len(deriv) == 0: return "Indefinido"
                max_deriv = np.max(np.abs(deriv))
                
                if max_deriv > 1e7: return "Diverge (Derivada no acotada)"
                val = ((b - a)**5 / (180 * n**4)) if metodo == "simpson13" else ((b - a)**5 / (80 * n**4))
                ans = round(float(val * max_deriv), precision)
            
            f.grabar_rescates = estado_previo
            return str(ans)
        except Exception:
            return "Indefinido matemáticamente"

    @staticmethod
    def rectangulo_compuesto(f: Callable, a: float, b: float, n: int, precision: int = 8, epsilon: float = None) -> Dict:
        IntegrationService._validar_integral(f, a, b)
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        x_medio = np.linspace(a + h/2, b - h/2, n)
        
        f.limites_calculados = []
        f.grabar_rescates = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_medio = f(x_medio)
        f.grabar_rescates = False
            
        IntegrationService._verificar_asintota(y_medio)
        integral = h * np.sum(y_medio)
        cota_error = IntegrationService._calcular_cota_error(f, a, b, n, "rectangulo", precision)
        et_puntual = IntegrationService._calcular_error_truncamiento_epsilon(f, a, b, n, "rectangulo", epsilon, precision)
        
        tabla = [{"i": 0, "x_n": round(float(x[0]), precision), "x_medio": None, "f_x_medio": None}]
        for i in range(n):
            tabla.append({"i": i + 1, "x_n": round(float(x[i+1]), precision), "x_medio": round(float(x_medio[i]), precision), "f_x_medio": round(float(y_medio[i]), precision)})
        
        intermedios = " + ".join([f"{yi:.{precision}f}" for yi in y_medio])
        desarrollo = f"I ≈ {h:.4f} [ {intermedios} ]"
        
        rescates_crudos = getattr(f, 'limites_calculados', [])
        notas_unicas = list(dict.fromkeys([f"Indeterminación en x = {pt:.4f}. Se aplicó Límite: f(x) = {val:.4f}" for pt, val in rescates_crudos]))
        
        return {
            "metodo": "Rectangulo Compuesto", "a": a, "b": b, "n": n, "h": round(h, precision),
            "integral": round(float(integral), precision), "cota_error": cota_error,
            "error_truncamiento": et_puntual["error_truncamiento"] if isinstance(et_puntual, dict) else et_puntual,
            "detalle_error_truncamiento": et_puntual if isinstance(et_puntual, dict) else None,
            "tabla": tabla, "desarrollo": desarrollo, "notas": notas_unicas
        }
    
    @staticmethod
    def trapecio_compuesto(f: Callable, a: float, b: float, n: int, precision: int = 8, epsilon: float = None) -> Dict:
        IntegrationService._validar_integral(f, a, b)
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        
        f.limites_calculados = []
        f.grabar_rescates = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = f(x)
        f.grabar_rescates = False
            
        IntegrationService._verificar_asintota(y)
        S = y[0] + y[-1] + 2 * np.sum(y[1:-1])
        integral = (h / 2) * S
        cota_error = IntegrationService._calcular_cota_error(f, a, b, n, "trapecio", precision)
        et_puntual = IntegrationService._calcular_error_truncamiento_epsilon(f, a, b, n, "trapecio", epsilon, precision)
        
        tabla = [{"i": i, "x_n": round(float(x[i]), precision), "f_x_n": round(float(y[i]), precision)} for i in range(len(x))]
        desarrollo = f"I ≈ {h:.4f}/2 [ {y[0]:.{precision}f} + {y[-1]:.{precision}f} ]" if n == 1 else f"I ≈ {h:.4f}/2 [ {y[0]:.{precision}f} + 2({' + '.join([f'{yi:.{precision}f}' for yi in y[1:-1]])}) + {y[-1]:.{precision}f} ]"
        
        rescates_crudos = getattr(f, 'limites_calculados', [])
        notas_unicas = list(dict.fromkeys([f"Indeterminación en x = {pt:.4f}. Se aplicó Límite: f(x) = {val:.4f}" for pt, val in rescates_crudos]))
        
        return {
            "metodo": "Trapecio Compuesto" if n > 1 else "Trapecio Simple", "a": a, "b": b, "n": n, "h": round(h, precision),
            "integral": round(float(integral), precision), "cota_error": cota_error,
            "error_truncamiento": et_puntual["error_truncamiento"] if isinstance(et_puntual, dict) else et_puntual,
            "detalle_error_truncamiento": et_puntual if isinstance(et_puntual, dict) else None,
            "tabla": tabla, "desarrollo": desarrollo, "notas": notas_unicas
        }
    
    @staticmethod
    def simpson_13_compuesto(f: Callable, a: float, b: float, n: int, precision: int = 8, epsilon: float = None) -> Dict:
        IntegrationService._validar_integral(f, a, b)
        if n % 2 != 0: raise ValueError("n debe ser PAR para Simpson 1/3")
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        
        f.limites_calculados = []
        f.grabar_rescates = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = f(x)
        f.grabar_rescates = False
            
        IntegrationService._verificar_asintota(y)
        impares = y[1:n:2]
        pares = y[2:n-1:2]
        
        S = y[0] + y[-1] + 4 * np.sum(impares) + 2 * np.sum(pares)
        integral = (h / 3) * S
        cota_error = IntegrationService._calcular_cota_error(f, a, b, n, "simpson13", precision)
        et_puntual = IntegrationService._calcular_error_truncamiento_epsilon(f, a, b, n, "simpson13", epsilon, precision)
        
        tabla = [{"i": i, "x_n": round(float(x[i]), precision), "f_x_n": round(float(y[i]), precision)} for i in range(len(x))]
        desarrollo = f"I ≈ {h:.4f}/3 [ {y[0]:.{precision}f} + 4({y[1]:.{precision}f}) + {y[2]:.{precision}f} ]" if n == 2 else f"I ≈ {h:.4f}/3 [ {y[0]:.{precision}f} + 4({' + '.join([f'{yi:.{precision}f}' for yi in impares])}) + 2({' + '.join([f'{yi:.{precision}f}' for yi in pares])}) + {y[-1]:.{precision}f} ]"
            
        rescates_crudos = getattr(f, 'limites_calculados', [])
        notas_unicas = list(dict.fromkeys([f"Indeterminación en x = {pt:.4f}. Se aplicó Límite: f(x) = {val:.4f}" for pt, val in rescates_crudos]))
        
        return {
            "metodo": "Simpson 1/3 Compuesto" if n > 2 else "Simpson 1/3 Simple", "a": a, "b": b, "n": n, "h": round(h, precision),
            "integral": round(float(integral), precision), "cota_error": cota_error,
            "error_truncamiento": et_puntual["error_truncamiento"] if isinstance(et_puntual, dict) else et_puntual,
            "detalle_error_truncamiento": et_puntual if isinstance(et_puntual, dict) else None,
            "tabla": tabla, "desarrollo": desarrollo, "notas": notas_unicas
        }
    
    @staticmethod
    def simpson_38_compuesto(f: Callable, a: float, b: float, n: int, precision: int = 8, epsilon: float = None) -> Dict:
        IntegrationService._validar_integral(f, a, b)
        if n % 3 != 0: raise ValueError("n debe ser multiplo de 3 para Simpson 3/8")
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        
        f.limites_calculados = []
        f.grabar_rescates = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = f(x)
        f.grabar_rescates = False
            
        IntegrationService._verificar_asintota(y)
        grupo_1, grupo_2, grupo_3 = y[1:n:3], y[2:n:3], y[3:n-1:3]
        
        S = y[0] + y[-1] + 3 * (np.sum(grupo_1) + np.sum(grupo_2)) + 2 * np.sum(grupo_3)
        integral = (3 * h / 8) * S
        cota_error = IntegrationService._calcular_cota_error(f, a, b, n, "simpson38", precision)
        et_puntual = IntegrationService._calcular_error_truncamiento_epsilon(f, a, b, n, "simpson38", epsilon, precision)
        
        tabla = [{"i": i, "x_n": round(float(x[i]), precision), "f_x_n": round(float(y[i]), precision)} for i in range(len(x))]
        if n == 3: desarrollo = f"I ≈ 3({h:.4f})/8 [ {y[0]:.{precision}f} + 3({y[1]:.{precision}f} + {y[2]:.{precision}f}) + {y[3]:.{precision}f} ]"
        else: desarrollo = f"I ≈ 3({h:.4f})/8 [ {y[0]:.{precision}f} + 3({' + '.join([f'{yi:.{precision}f}' for yi in np.concatenate((grupo_1, grupo_2))])}){(' + 2(' + ' + '.join([f'{yi:.{precision}f}' for yi in grupo_3]) + ')') if len(grupo_3) > 0 else ''} + {y[-1]:.{precision}f} ]"
            
        rescates_crudos = getattr(f, 'limites_calculados', [])
        notas_unicas = list(dict.fromkeys([f"Indeterminación en x = {pt:.4f}. Se aplicó Límite: f(x) = {val:.4f}" for pt, val in rescates_crudos]))
        
        return {
            "metodo": "Simpson 3/8 Compuesto" if n > 3 else "Simpson 3/8 Simple", "a": a, "b": b, "n": n, "h": round(h, precision),
            "integral": round(float(integral), precision), "cota_error": cota_error,
            "error_truncamiento": et_puntual["error_truncamiento"] if isinstance(et_puntual, dict) else et_puntual,
            "detalle_error_truncamiento": et_puntual if isinstance(et_puntual, dict) else None,
            "tabla": tabla, "desarrollo": desarrollo, "notas": notas_unicas
        }
    @staticmethod
    def comparar_metodos(func_str: str, a: float, b: float, n: int, precision: int = 8, epsilon: float = None) -> Dict:
        """
        Orquestador que ejecuta los 4 métodos de integración y devuelve un resumen comparativo.
        """
        resultados = {
            "rectangulo": None,
            "trapecio": None,
            "simpson_13": None,
            "simpson_38": None
        }

        try:
            f = IntegrationService.compilar_funcion(func_str)
        except Exception as e:
            return {"error_global": str(e)}

        # 1. Rectángulo Medio Compuesto
        try:
            resultados["rectangulo"] = IntegrationService.rectangulo_compuesto(f, a, b, n, precision, epsilon)
            resultados["rectangulo"]["exito"] = True
        except Exception as e:
            resultados["rectangulo"] = {"exito": False, "error_msg": str(e)}

        # 2. Trapecio Compuesto
        try:
            resultados["trapecio"] = IntegrationService.trapecio_compuesto(f, a, b, n, precision, epsilon)
            resultados["trapecio"]["exito"] = True
        except Exception as e:
            resultados["trapecio"] = {"exito": False, "error_msg": str(e)}

        # 3. Simpson 1/3 Compuesto
        try:
            resultados["simpson_13"] = IntegrationService.simpson_13_compuesto(f, a, b, n, precision, epsilon)
            resultados["simpson_13"]["exito"] = True
        except Exception as e:
            resultados["simpson_13"] = {"exito": False, "error_msg": str(e)}

        # 4. Simpson 3/8 Compuesto
        try:
            resultados["simpson_38"] = IntegrationService.simpson_38_compuesto(f, a, b, n, precision, epsilon)
            resultados["simpson_38"]["exito"] = True
        except Exception as e:
            resultados["simpson_38"] = {"exito": False, "error_msg": str(e)}

        return {
            "parametros": {"f(x)": func_str, "a": a, "b": b, "n": n, "epsilon": epsilon},
            "comparativa": resultados
        }