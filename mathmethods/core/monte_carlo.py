import numpy as np
import sympy as sp
import warnings
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss
from typing import Dict, Tuple, List
from .utils import safe_sympify

class MonteCarloService:
    @staticmethod
    def aplicar_semilla(seed=None):
        if seed is not None:
            np.random.seed(int(seed))
        else:
            np.random.seed(None)

    @staticmethod
    def calcular_z_score(nivel_confianza=0.95):
        alfa = 1.0 - nivel_confianza
        z_score = norm.ppf(1.0 - (alfa / 2.0))
        return z_score

    @staticmethod
    def compilar_funcion(texto_funcion: str, variables: str = 'x'):
        try:
            vars_sympy = sp.symbols(variables)
            expr = safe_sympify(texto_funcion, allowed_symbols=variables.split())
            f_compilada = sp.lambdify(vars_sympy, expr, modules=['numpy'])
            return f_compilada
        except Exception as e:
            raise ValueError(f"Error compilando funcion: {str(e)}")

    @staticmethod
    def hit_or_miss_1d(f, a: float, b: float, N: int, seed: int = None, precision: int = 8, nivel_confianza: float = 0.95) -> Dict:
        MonteCarloService.aplicar_semilla(seed)
        
        # Le pedimos a Python que calcule los puntos de la línea para el gráfico
        x_test = np.linspace(a, b, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_test_raw = f(x_test)

        # Detección de singularidades en el barrido (para no silenciarlas)
        con_singularidades = bool(np.any(~np.isfinite(y_test_raw)))
        y_test = np.nan_to_num(y_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

        y_min, y_max = float(np.min(y_test)), float(np.max(y_test))
        y_base = min(0, y_min) * 1.1 if min(0, y_min) < 0 else 0
        y_techo = max(0, y_max) * 1.1

        x_rand = np.random.uniform(a, b, N)
        y_rand = np.random.uniform(y_base, y_techo, N)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = f(x_rand)

        exitos_pos = (y_rand > 0) & (y_rand <= f_eval)
        exitos_neg = (y_rand < 0) & (y_rand >= f_eval)
        exitos_totales = exitos_pos | exitos_neg
        n_exitos = int(np.sum(exitos_totales))
        n_pos = int(np.sum(exitos_pos))
        n_neg = int(np.sum(exitos_neg))

        # Estimación con signo: la caja total [y_base, y_techo] contiene la región
        # positiva y la negativa. El estimador de Hit-or-Miss con signo es
        #   I ≈ (b - a) * (y_techo - y_base) * (n_pos - n_neg) / N
        # que reproduce ∫f⁺ - ∫f⁻ (antes devolvía ±∫|f| para integrandos con signo).
        area_caja = (b - a) * (y_techo - y_base)
        integral_aprox = area_caja * (n_pos - n_neg) / N

        n_no_finitos = int(np.sum(~np.isfinite(f_eval)))
        p_exito = n_exitos / N
        notas = []
        if con_singularidades:
            notas.append(
                "Se detectaron singularidades en el barrido de la caja; los límites se "
                "calcularon con los valores finitos."
            )
        if n_no_finitos > 0:
            notas.append(
                f"{n_no_finitos} muestra(s) no finita(s) se descartaron; el resultado puede estar sesgado."
            )

        # Para hit-or-miss, la media muestral natural es la media del indicador con signo.
        media_muestral = (n_pos - n_neg) / N

        # Cálculo Estadístico para Hit-or-Miss (Distribución Binomial)
        desv_std = np.sqrt(p_exito * (1 - p_exito)) if N > 1 else 0
        error_est = desv_std / np.sqrt(N) if N > 0 else 0
        
        z_score = MonteCarloService.calcular_z_score(nivel_confianza)
        ic_inf = integral_aprox - z_score * error_est * area_caja
        ic_sup = integral_aprox + z_score * error_est * area_caja

        # Capeamos a 2000 y ponemos los tildes
        limite = min(N, 2000)
        historial = [
            {"i": i+1, "x": round(float(x_rand[i]), precision), "y_rand": round(float(y_rand[i]), precision), 
             "f_x": round(float(f_eval[i]), precision), "exito": "✓" if exitos_totales[i] else "✗"} 
            for i in range(limite)
        ]
        
        return {
            "metodo": "Hit-or-Miss 1D",
            "integral": round(float(integral_aprox), precision),
            "media_muestral": round(float(media_muestral), precision),
            "N": N,
            "area_caja": round(float(area_caja), precision),
            "n_exitos": n_exitos,
            "escala": round(float(area_caja), precision),
            "desv_estandar": round(float(desv_std), precision),
            "error_std": round(float(error_est), precision),
            "z_score": round(float(z_score), 4),
            "ic_inf": round(float(ic_inf), precision),
            "ic_sup": round(float(ic_sup), precision),
            "historial": historial,
            "x_line": [round(float(x), precision) for x in x_test], # Línea para el frontend
            "y_line": [round(float(y), precision) for y in y_test],
            "grafico_limites": {"y_base": float(y_base), "y_techo": float(y_techo)},
            "notas": notas
        }


    @staticmethod
    def convergencia_1d(f, a: float, b: float, N: int, seed: int = None, precision: int = 8) -> Dict:
        MonteCarloService.aplicar_semilla(seed)
        x_rand = np.random.uniform(a, b, N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = np.nan_to_num(f(x_rand), nan=0.0)
            
        promedios_acumulados = np.cumsum(f_eval) / np.arange(1, N + 1)
        integrales_acumuladas = (b - a) * promedios_acumulados

        # Gauss-Legendre para la línea base de comparación exacta
        x_g, w_g = leggauss(5)
        x_trans = 0.5 * (x_g + 1) * (b - a) + a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_gauss = f(x_trans)
        valor_exacto = 0.5 * (b - a) * np.sum(w_g * f_gauss)
        
        # Muestreamos el historial logarítmicamente para no matar a Plotly
        step = max(1, N // 500)
        historial = [
            {"N": int(i), "integral": round(float(integrales_acumuladas[i-1]), precision)}
            for i in range(1, N+1, step)
        ]

        return {
            "metodo": "Convergencia 1D",
            "N": N,
            "valor_exacto_gauss": round(float(valor_exacto), precision),
            "historial_convergencia": historial
        }

    @staticmethod
    def valor_promedio_1d(f, a: float, b: float, N: int, seed: int = None, precision: int = 8, nivel_confianza: float = 0.95) -> Dict:
        MonteCarloService.aplicar_semilla(seed)
        x_rand = np.random.uniform(a, b, N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = f(x_rand)
            f_eval = np.nan_to_num(f_eval, nan=0.0, posinf=0.0, neginf=0.0)
        
        promedio = np.mean(f_eval)
        escala = b - a
        integral = escala * promedio
        
        desv_std = np.std(f_eval, ddof=1) if N > 1 else 0
        error_est = desv_std / np.sqrt(N) if N > 0 else 0
        
        z_score = MonteCarloService.calcular_z_score(nivel_confianza)
        ic_inf = integral - z_score * error_est * escala
        ic_sup = integral + z_score * error_est * escala
        
        limite = min(N, 2000)
        historial = [{"i": i+1, "x": round(float(x_rand[i]), precision), "f_x": round(float(f_eval[i]), precision)} for i in range(limite)]
        
        return {
            "metodo": "Valor Promedio 1D", "integral": round(float(integral), precision),
            "media_muestral": round(float(promedio), precision),
            "promedio_fx": round(float(promedio), precision), "escala": round(float(escala), precision),
            "desv_estandar": round(float(desv_std), precision), "error_std": round(float(error_est), precision),
            "z_score": round(float(z_score), 4), "ic_inf": round(float(ic_inf), precision),
            "ic_sup": round(float(ic_sup), precision), "N": N, "historial": historial
        }

    @staticmethod
    def valor_promedio_2d(f, lim_x: Tuple[float, float], lim_y: Tuple[float, float], N: int, seed: int = None, precision: int = 8, nivel_confianza: float = 0.95) -> Dict:
        MonteCarloService.aplicar_semilla(seed)
        ax, bx = lim_x; ay, by = lim_y
        
        x_rand = np.random.uniform(ax, bx, N)
        y_rand = np.random.uniform(ay, by, N)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = f(x_rand, y_rand)
            f_eval = np.nan_to_num(f_eval, nan=0.0, posinf=0.0, neginf=0.0)
            
        escala = (bx - ax) * (by - ay)
        promedio = np.mean(f_eval)
        integral = escala * promedio
        
        desv_std = np.std(f_eval, ddof=1) if N > 1 else 0
        error_est = desv_std / np.sqrt(N) if N > 0 else 0
        
        z_score = MonteCarloService.calcular_z_score(nivel_confianza)
        ic_inf = integral - z_score * error_est * escala
        ic_sup = integral + z_score * error_est * escala

        limite = min(N, 2000)
        historial = [{"i": i+1, "x": round(float(x_rand[i]), precision), "y": round(float(y_rand[i]), precision), "f_xy": round(float(f_eval[i]), precision)} for i in range(limite)]
        
        return {
            "metodo": "Valor Promedio 2D", "integral": round(float(integral), precision),
            "media_muestral": round(float(promedio), precision),
            "promedio_fxy": round(float(promedio), precision), "escala": round(float(escala), precision),
            "desv_estandar": round(float(desv_std), precision), "error_std": round(float(error_est), precision),
            "z_score": round(float(z_score), 4), "ic_inf": round(float(ic_inf), precision),
            "ic_sup": round(float(ic_sup), precision), "N": N, "historial": historial
        }

    @staticmethod
    def valor_promedio_3d(f, lim_x: Tuple[float, float], lim_y: Tuple[float, float], lim_z: Tuple[float, float], N: int, seed: int = None, precision: int = 8, nivel_confianza: float = 0.95) -> Dict:
        MonteCarloService.aplicar_semilla(seed)
        ax, bx = lim_x; ay, by = lim_y; az, bz = lim_z
        
        x_rand = np.random.uniform(ax, bx, N)
        y_rand = np.random.uniform(ay, by, N)
        z_rand = np.random.uniform(az, bz, N)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = f(x_rand, y_rand, z_rand)
            f_eval = np.nan_to_num(f_eval, nan=0.0, posinf=0.0, neginf=0.0)
            
        escala = (bx - ax) * (by - ay) * (bz - az)
        promedio = np.mean(f_eval)
        integral = escala * promedio
        
        desv_std = np.std(f_eval, ddof=1) if N > 1 else 0
        error_est = desv_std / np.sqrt(N) if N > 0 else 0
        z_score = MonteCarloService.calcular_z_score(nivel_confianza)
        ic_inf = integral - z_score * error_est * escala
        ic_sup = integral + z_score * error_est * escala

        limite = min(N, 2000)
        historial = [{"i": i+1, "x": round(float(x_rand[i]), precision), "y": round(float(y_rand[i]), precision), "z": round(float(z_rand[i]), precision), "f_xyz": round(float(f_eval[i]), precision)} for i in range(limite)]
        
        return {
            "metodo": "Valor Promedio 3D", "integral": round(float(integral), precision),
            "media_muestral": round(float(promedio), precision),
            "promedio_fxyz": round(float(promedio), precision), "escala": round(float(escala), precision),
            "desv_estandar": round(float(desv_std), precision), "error_std": round(float(error_est), precision),
            "z_score": round(float(z_score), 4), "ic_inf": round(float(ic_inf), precision),
            "ic_sup": round(float(ic_sup), precision), "N": N, "historial": historial
        }

    @staticmethod
    def analisis_estadistico_1d(f, a: float, b: float, N: int, M: int, nivel_confianza: float = 0.95, seed: int = None, precision: int = 8) -> Dict:
        MonteCarloService.aplicar_semilla(seed)
        
        x_rand = np.random.uniform(a, b, (M, N))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_eval = np.nan_to_num(f(x_rand), nan=0.0)
            
        integrales = (b - a) * np.mean(f_eval, axis=1)
        
        media = np.mean(integrales)
        varianza = np.var(integrales, ddof=1)
        desviacion = np.std(integrales, ddof=1)
        
        z_score = MonteCarloService.calcular_z_score(nivel_confianza)
        error_estandar = desviacion / np.sqrt(M)
        ic_inf = media - z_score * error_estandar
        ic_sup = media + z_score * error_estandar
        
        return {
            "metodo": "Estadístico 1D",
            "media": round(float(media), precision),
            "integral": round(float(media), precision), # Alias para frontend
            "varianza": round(float(varianza), precision),
            "desv_estandar": round(float(desviacion), precision),
            "error_std": round(float(error_estandar), precision),
            "z_score": round(float(z_score), 4),
            "ic_inf": round(float(ic_inf), precision),
            "ic_sup": round(float(ic_sup), precision),
            "ancho_ic": round(float(ic_sup - ic_inf), precision),
            "nivel_confianza": nivel_confianza,
            "M": M,
            "N": N,
            "distribucion": [round(float(val), precision) for val in integrales]
        }