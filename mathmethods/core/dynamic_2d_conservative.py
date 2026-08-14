# backend/app/methods/dynamic_2d_conservative.py
import math
from typing import Any, Dict

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)
from .utils import safe_parse_expr


class Dynamic2DConservativeService:
    @staticmethod
    def clasificar_punto(traza, determinante) -> tuple:
        if abs(determinante) < 1e-10:
            return "Punto Degenerado", "Jacobiano singular. Requiere análisis superior."
        elif determinante < 0:
            return "Silla", "Punto hiperbólico inestable. Por aquí pasa la Separatriz."
        # determinante > 0
        if abs(traza) < 1e-10:
            return (
                "Centro",
                "Punto estable (no asintótico). Las trayectorias forman órbitas cerradas (conservación de energía).",
            )
        discriminante = traza**2 - 4 * determinante
        if discriminante < 0:
            if traza < 0:
                return "Foco Estable", "Espiral convergente (atractor). Las trayectorias se enrollan hacia el punto."
            return "Foco Inestable", "Espiral divergente (repulsor). Las trayectorias se alejan del punto."
        if traza < 0:
            return "Nodo Estable", "Nodo asintóticamente estable. Las trayectorias convergen al punto."
        return "Nodo Inestable", "Nodo repulsivo. Las trayectorias divergen del punto."

    @staticmethod
    def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
        eq_x_str = payload.get("eq_x", "y")
        eq_y_str = payload.get("eq_y", "x - x**3")
        mu_val = float(payload.get("mu", 0.0))
        x0 = float(payload.get("x0", 0.1))
        y0 = float(payload.get("y0", 0.0))
        t0 = float(payload.get("t0", 0.0))
        t_fin = float(payload.get("t_fin", 15.0))
        h = float(payload.get("h", 0.02))
        x_min = float(payload.get("x_min", -3.0))
        x_max = float(payload.get("x_max", 3.0))
        y_min = float(payload.get("y_min", -3.0))
        y_max = float(payload.get("y_max", 3.0))
        cantidad_trayectorias = int(payload.get("cantidad_trayectorias", 20))

        x, y, mu_sym = sp.symbols("x y mu", real=True)
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        try:
            f_raw = safe_parse_expr(
                eq_x_str,
                local_dict={"x": x, "y": y, "mu": mu_sym},
                allowed_symbols=["x", "y", "mu"],
                transformations=transformations,
            )
            g_raw = safe_parse_expr(
                eq_y_str,
                local_dict={"x": x, "y": y, "mu": mu_sym},
                allowed_symbols=["x", "y", "mu"],
                transformations=transformations,
            )

            f = f_raw.subs(mu_sym, mu_val)
            g = g_raw.subs(mu_sym, mu_val)
        except Exception as e:
            raise ValueError(f"Error de sintaxis: {str(e)}")

        # 1. Calculo de Divergencia
        divergencia = sp.diff(f, x) + sp.diff(g, y)
        div_simplify = sp.simplify(divergencia)
        es_conservativo = div_simplify == 0

        # 2. Integracion del Hamiltoniano / Energia E(x,y) y Demostracion dH/dt
        H_latex = ""
        H_num = None
        dh_dx_latex = ""
        dh_dy_latex = ""
        dh_dt_raw_latex = ""
        dh_dt_simplified_latex = ""

        if es_conservativo:
            try:
                # Ecuaciones de Hamilton: dH/dy = f  y  dH/dx = -g
                H_y = sp.integrate(f, y)
                rem = -g - sp.diff(H_y, x)
                H_x = sp.integrate(rem, x)
                H = sp.simplify(H_y + H_x)
                H_latex = sp.latex(H)
                H_num = sp.lambdify((x, y), H, "numpy")

                # Demostracion dH/dt = (dH/dx)*x_dot + (dH/dy)*y_dot
                dH_dx = sp.diff(H, x)
                dH_dy = sp.diff(H, y)
                dh_dt_raw = dH_dx * f + dH_dy * g
                dh_dt_simplified = sp.simplify(dh_dt_raw)

                dh_dx_latex = sp.latex(dH_dx)
                dh_dy_latex = sp.latex(dH_dy)
                dh_dt_raw_latex = sp.latex(dh_dt_raw)
                dh_dt_simplified_latex = sp.latex(dh_dt_simplified)
            except:
                H_latex = "\\text{No integrable simbólicamente}"

        # 3. Matriz Jacobiana Analitica
        J = sp.Matrix([[sp.diff(f, x), sp.diff(f, y)], [sp.diff(g, x), sp.diff(g, y)]])

        # 4. Puntos de Equilibrio
        try:
            sols = sp.solve([f, g], (x, y), dict=True)
            puntos_equilibrio = []
            for sol in sols:
                if x in sol and y in sol:
                    x_val, y_val = complex(sol[x]), complex(sol[y])
                    if abs(x_val.imag) < 1e-9 and abs(y_val.imag) < 1e-9:
                        pto = (float(x_val.real), float(y_val.real))
                        if pto not in puntos_equilibrio:
                            puntos_equilibrio.append(pto)
        except:
            puntos_equilibrio = []

        analisis_puntos = []
        saddle_energies = []

        for pto in puntos_equilibrio:
            px, py = pto[0], pto[1]
            J_eval = J.subs({x: px, y: py})
            J_num_matrix = np.array(J_eval.tolist(), dtype=float)

            traza = float(np.trace(J_num_matrix))
            determinante = float(np.linalg.det(J_num_matrix))
            discriminante = traza**2 - 4 * determinante
            autovalores, _ = np.linalg.eig(J_num_matrix)

            clasif, comp = Dynamic2DConservativeService.clasificar_punto(
                traza, determinante
            )

            if "Silla" in clasif and H_num is not None:
                try:
                    saddle_energies.append(float(H_num(px, py)))
                except:
                    pass

            j_local_formateado = [
                [
                    round(float(J_num_matrix[0][0]), 4),
                    round(float(J_num_matrix[0][1]), 4),
                ],
                [
                    round(float(J_num_matrix[1][0]), 4),
                    round(float(J_num_matrix[1][1]), 4),
                ],
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
                    "autovalores": [
                        {"real": float(np.real(av)), "imag": float(np.imag(av))}
                        for av in autovalores
                    ],
                    "jacobiano_local": j_local_formateado,
                }
            )

        # 5. Simulacion Numerica (RK4)
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
            X_main.append(rk4_step(curr, h))
        X_main_final = np.array([p for p in X_main if p is not None])

        automatic_trajectories = []
        n_side = int(math.ceil(math.sqrt(cantidad_trayectorias)))
        if n_side > 1:
            x0_auto = np.linspace(x_min, x_max, n_side)
            y0_auto = np.linspace(y_min, y_max, n_side)
            for xi in x0_auto:
                for yi in y0_auto:
                    X_temp = [np.array([xi, yi])]
                    for _ in range(int(len(t_vals) / 2) - 1):
                        curr = X_temp[-1]
                        if curr is None:
                            break
                        X_temp.append(rk4_step(curr, h * 2))

                    X_temp_final = np.array([p for p in X_temp if p is not None])
                    if len(X_temp_final) > 1:
                        automatic_trajectories.append(
                            {
                                "x": X_temp_final[:, 0].tolist(),
                                "y": X_temp_final[:, 1].tolist(),
                            }
                        )

        # 6. Grillas para Plotly
        grid_res = 150
        X_mesh, Y_mesh = np.meshgrid(
            np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res)
        )

        Z_H_grid = np.zeros_like(X_mesh)
        if H_num is not None:
            try:
                Z_H_grid = H_num(X_mesh, Y_mesh)
                if np.isscalar(Z_H_grid):
                    Z_H_grid = np.full(X_mesh.shape, Z_H_grid)
                Z_H_grid = np.nan_to_num(Z_H_grid, nan=0.0, posinf=1e5, neginf=-1e5)
                Z_H_grid = np.clip(Z_H_grid, -1e6, 1e6)
            except:
                pass

        return {
            "ecuacion_latex_x": sp.latex(f_raw),
            "ecuacion_latex_y": sp.latex(g_raw),
            "divergencia_latex": sp.latex(div_simplify),
            "es_conservativo": bool(es_conservativo),
            "hamiltoniano_latex": H_latex,
            "dh_dx_latex": dh_dx_latex,
            "dh_dy_latex": dh_dy_latex,
            "dh_dt_raw_latex": dh_dt_raw_latex,
            "dh_dt_simplified_latex": dh_dt_simplified_latex,
            "saddle_energies": list(set(saddle_energies)),
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
                "z_H": Z_H_grid.tolist(),
            },
        }
