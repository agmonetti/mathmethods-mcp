# backend/app/methods/dynamic_2d_nonlinear.py
import math
from typing import Any, Dict

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)
from .utils import safe_parse_expr


class Dynamic2DNonLinearService:
    @staticmethod
    def clasificar_punto(traza, determinante, discriminante, autovalores) -> tuple:
        if abs(determinante) < 1e-10:
            clasif = "Punto Degenerado (No Hiperbólico)"
            comp = "El Jacobiano es singular (al menos un autovalor es cero). Falla la linealización. Esto suele indicar una variedad central donde ocurren bifurcaciones estáticas (Silla-Nodo, Transcrítica o Pitchfork)."
            return clasif, comp

        elif determinante < 0:
            clasif = "Silla"
            comp = "Trayectorias se acercan por una dirección (variedad estable) y se alejan por otra (variedad inestable). Punto hiperbólico."

        elif discriminante > 1e-10:
            if traza < 0:
                clasif = "Nodo Estable"
                comp = "Trayectorias tienden al equilibrio asintóticamente sin oscilar (Sumidero puro). Punto hiperbólico."
            elif traza > 0:
                clasif = "Nodo Inestable"
                comp = "Trayectorias se alejan del equilibrio sin oscilar (Fuente pura). Punto hiperbólico."
            else:
                clasif = "Nodo con traza cero (Degenerado)"
                comp = "Comportamiento nodal limítrofe."

        elif abs(discriminante) <= 1e-10:
            if traza < 0:
                clasif = "Nodo Degenerado Estable"
                comp = "Trayectorias convergen al punto, con un autovector dominante. Punto hiperbólico."
            elif traza > 0:
                clasif = "Nodo Degenerado Inestable"
                comp = "Trayectorias divergen del punto, con un autovector dominante. Punto hiperbólico."
            else:
                clasif = "Caso No Hiperbólico"
                comp = "Falla el teorema de Hartman-Grobman."

        else:
            parte_real = float(np.real(autovalores[0]))
            if abs(parte_real) < 1e-10:
                clasif = "Centro o Bifurcación de Hopf (No Hiperbólico)"
                comp = "Los autovalores son puramente imaginarios. Hartman-Grobman falla. Físicamente, el punto puede ser un 'Centro' conservativo, o estar en el punto crítico de una Bifurcación de Hopf (nacimiento/muerte de un Ciclo Límite). Confirmar con Funciones de Lyapunov o cambio a Coordenadas Polares."
            elif parte_real < 0:
                clasif = "Foco Estable"
                comp = "Trayectorias giran en espiral convergiendo hacia el equilibrio (Atractor). Punto hiperbólico."
            else:
                clasif = "Foco Inestable"
                comp = "Trayectorias giran en espiral alejándose del equilibrio (Repulsor). Punto hiperbólico."

        return clasif, comp

    @staticmethod
    def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
        eq_x_str = payload.get("eq_x", "y - x")
        eq_y_str = payload.get("eq_y", "x**2 - 1")
        params = payload.get("params", {}) or {}
        params = {str(k): float(v) for k, v in params.items()}
        mu_val = float(params.pop("mu", payload.get("mu", 0.0)))
        x0 = float(payload.get("x0", 0.5))
        y0 = float(payload.get("y0", 0.5))
        t0 = float(payload.get("t0", 0.0))
        t_fin = float(payload.get("t_fin", 10.0))
        h = float(payload.get("h", 0.02))
        x_min = float(payload.get("x_min", -4.0))
        x_max = float(payload.get("x_max", 4.0))
        y_min = float(payload.get("y_min", -4.0))
        y_max = float(payload.get("y_max", 4.0))
        cantidad_trayectorias = int(payload.get("cantidad_trayectorias", 25))

        x, y, mu_sym = sp.symbols("x y mu", real=True)
        param_symbols = {name: sp.Symbol(name, real=True) for name in params.keys()}
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        try:
            local_dict = {"x": x, "y": y, "mu": mu_sym, **param_symbols}
            gate_allowed = list(dict.fromkeys(["x", "y", "mu"] + list(params.keys())))
            f_raw = safe_parse_expr(
                eq_x_str,
                local_dict=local_dict,
                allowed_symbols=gate_allowed,
                transformations=transformations,
            )
            g_raw = safe_parse_expr(
                eq_y_str,
                local_dict=local_dict,
                allowed_symbols=gate_allowed,
                transformations=transformations,
            )

            subs_map = {mu_sym: mu_val, **{param_symbols[k]: v for k, v in params.items()}}
            f = f_raw.subs(subs_map)
            g = g_raw.subs(subs_map)

            free_symbols = {str(sym) for sym in f.free_symbols.union(g.free_symbols)}
            missing = sorted(sym for sym in free_symbols if sym not in {"x", "y"})
            if missing:
                raise ValueError(f"Faltan parametros: {', '.join(missing)}")
        except Exception as e:
            raise ValueError(f"Error de sintaxis en las ecuaciones: {str(e)}")

        J = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)], [sp.diff(g, x), sp.diff(g, y)]])

        sols = sp.nonlinsolve([f, g], [x, y])
        puntos_equilibrio = []

        if isinstance(sols, sp.FiniteSet):
            for sol in sols:
                try:
                    x_val = complex(sol[0])
                    y_val = complex(sol[1])
                    if abs(x_val.imag) < 1e-9 and abs(y_val.imag) < 1e-9:
                        puntos_equilibrio.append((x_val.real, y_val.real))
                except:
                    pass

        analisis_puntos = []
        for pto in puntos_equilibrio:
            px, py = pto[0], pto[1]
            J_eval = J.subs({x: px, y: py})
            J_num = np.array(J_eval.tolist(), dtype=float)

            traza = float(np.trace(J_num))
            determinante = float(np.linalg.det(J_num))
            discriminante = traza**2 - 4 * determinante
            autovalores, autovectores = np.linalg.eig(J_num)

            clasif, comp = Dynamic2DNonLinearService.clasificar_punto(
                traza, determinante, discriminante, autovalores
            )
            autos_formateados = [
                {"real": float(np.real(av)), "imag": float(np.imag(av))}
                for av in autovalores
            ]

            j_local_formateado = [
                [round(float(J_num[0][0]), 4), round(float(J_num[0][1]), 4)],
                [round(float(J_num[1][0]), 4), round(float(J_num[1][1]), 4)],
            ]

            analisis_puntos.append(
                {
                    "x": px,
                    "y": py,
                    "traza": traza,
                    "determinante": determinante,
                    "discriminante": discriminante,
                    "clasificacion": clasif,
                    "comportamiento": comp,
                    "autovalores": autos_formateados,
                    "jacobiano_local": j_local_formateado,
                }
            )

        f_num = sp.lambdify((x, y), f, "numpy")
        g_num = sp.lambdify((x, y), g, "numpy")

        def rk4_step(X_curr, step_h):
            try:
                v_x, v_y = float(X_curr[0]), float(X_curr[1])
                k1_x = f_num(v_x, v_y)
                k1_y = g_num(v_x, v_y)

                k2_x = f_num(v_x + step_h * k1_x / 2, v_y + step_h * k1_y / 2)
                k2_y = g_num(v_x + step_h * k1_x / 2, v_y + step_h * k1_y / 2)

                k3_x = f_num(v_x + step_h * k2_x / 2, v_y + step_h * k2_y / 2)
                k3_y = g_num(v_x + step_h * k2_x / 2, v_y + step_h * k2_y / 2)

                k4_x = f_num(v_x + step_h * k3_x, v_y + step_h * k3_y)
                k4_y = g_num(v_x + step_h * k3_x, v_y + step_h * k3_y)

                new_x = v_x + (step_h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                new_y = v_y + (step_h / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

                if (
                    math.isnan(new_x)
                    or math.isnan(new_y)
                    or abs(new_x) > 1e6
                    or abs(new_y) > 1e6
                ):
                    return None
                return np.array([new_x, new_y])
            except:
                return None

        n_steps_t = max(1, int(round((t_fin - t0) / h)))
        t_vals = np.linspace(t0, t_fin, n_steps_t + 1)
        X_main = [np.array([x0, y0])]
        for _ in range(len(t_vals) - 1):
            curr = X_main[-1]
            if curr is None:
                break
            next_p = rk4_step(curr, h)
            X_main.append(next_p)
        X_main_final = np.array([p for p in X_main if p is not None])

        automatic_trajectories = []
        n_side = int(math.sqrt(cantidad_trayectorias))
        if n_side > 1:
            x0_auto = np.linspace(x_min, x_max, n_side)
            y0_auto = np.linspace(y_min, y_max, n_side)
            for xi in x0_auto:
                for yi in y0_auto:
                    t_len = int(len(t_vals) / 2)
                    X_temp = [np.array([xi, yi])]
                    for _ in range(t_len - 1):
                        curr = X_temp[-1]
                        if curr is None:
                            break
                        next_p = rk4_step(curr, h * 2)
                        X_temp.append(next_p)

                    X_temp_final = np.array([p for p in X_temp if p is not None])
                    if len(X_temp_final) > 1:
                        automatic_trajectories.append(
                            {
                                "x": X_temp_final[:, 0].tolist(),
                                "y": X_temp_final[:, 1].tolist(),
                            }
                        )

        # --- ARREGLO DE NULCLINAS ROTA ---
        # Aumentamos la resolucion de la grilla a 150 para curvas ultra suaves
        grid_res = 150
        X_mesh, Y_mesh = np.meshgrid(
            np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res)
        )
        try:
            Z_x = f_num(X_mesh, Y_mesh)
            Z_y = g_num(X_mesh, Y_mesh)
            if np.isscalar(Z_x):
                Z_x = np.full(X_mesh.shape, Z_x)
            if np.isscalar(Z_y):
                Z_y = np.full(Y_mesh.shape, Z_y)

            # Recortamos los valores explosivos y limpiamos NaNs
            Z_x = np.nan_to_num(Z_x, nan=0.0, posinf=1e5, neginf=-1e5)
            Z_y = np.nan_to_num(Z_y, nan=0.0, posinf=1e5, neginf=-1e5)
            Z_x = np.clip(Z_x, -1e6, 1e6)
            Z_y = np.clip(Z_y, -1e6, 1e6)

        except Exception:
            Z_x = np.zeros_like(X_mesh)
            Z_y = np.zeros_like(Y_mesh)

        x_sym, y_sym_var = sp.symbols("x y", real=True)
        try:
            sol_x = sp.solve(sp.Eq(f, 0), y_sym_var)
            nul_x_despejada = (
                f"y = {str(sol_x[0]).replace('**', '^').replace('*', ' ')}"
                if sol_x
                else "No despejable como y(x)"
            )
        except:
            nul_x_despejada = "No despejable como y(x)"

        try:
            sol_y = sp.solve(sp.Eq(g, 0), y_sym_var)
            nul_y_despejada = (
                f"y = {str(sol_y[0]).replace('**', '^').replace('*', ' ')}"
                if sol_y
                else "No despejable como y(x)"
            )
        except:
            nul_y_despejada = "No despejable como y(x)"

        return {
            "ecuacion_latex_x": sp.latex(f_raw),
            "ecuacion_latex_y": sp.latex(g_raw),
            "jacobiano_latex": sp.latex(J),
            "puntos_analizados": analisis_puntos,
            "principal_trajectory": {
                "t": t_vals[: len(X_main_final)].tolist(),
                "x": X_main_final[:, 0].tolist(),
                "y": X_main_final[:, 1].tolist(),
            },
            "automatic_trajectories": automatic_trajectories,
            "contour_data": {
                "x_axis": X_mesh[0].tolist(),
                "y_axis": Y_mesh[:, 0].tolist(),
                "z_x_nul": Z_x.tolist(),
                "z_y_nul": Z_y.tolist(),
            },
            "nulclina_x_desp": nul_x_despejada,
            "nulclina_y_desp": nul_y_despejada,
        }
