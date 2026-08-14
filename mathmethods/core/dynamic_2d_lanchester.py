# backend/app/methods/dynamic_2d_lanchester.py
import math
from typing import Any, Dict

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
)
from .utils import safe_parse_expr


class Dynamic2DLanchesterService:
    @staticmethod
    def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
        eq_x_str = payload.get("eq_x", "-α * y")
        eq_y_str = payload.get("eq_y", "-β * x")
        a_val = float(payload.get("alpha", 1.0))
        b_val = float(payload.get("beta", 2.0))
        g_val = float(payload.get("gamma", 0.0))
        e_val = float(payload.get("epsilon", 0.0))
        m_val = float(payload.get("mu", 0.0))
        d_val = float(payload.get("delta", 0.0))

        x0 = float(payload.get("x0", 100.0))
        y0 = float(payload.get("y0", 80.0))
        t0 = float(payload.get("t0", 0.0))
        t_fin = float(payload.get("t_fin", 5.0))
        h = float(payload.get("h", 0.01))

        x, y = sp.symbols("x y", real=True)
        sym_alpha = sp.Symbol("α", real=True)
        sym_beta = sp.Symbol("β", real=True)
        sym_gamma = sp.Symbol("γ", real=True)
        sym_eps = sp.Symbol("ε", real=True)
        sym_mu = sp.Symbol("μ", real=True)
        sym_delta = sp.Symbol("δ", real=True)

        local_dict = {
            "x": x,
            "y": y,
            "α": sym_alpha,
            "β": sym_beta,
            "γ": sym_gamma,
            "ε": sym_eps,
            "μ": sym_mu,
            "δ": sym_delta,
        }
        subs_dict = {
            sym_alpha: a_val,
            sym_beta: b_val,
            sym_gamma: g_val,
            sym_eps: e_val,
            sym_mu: m_val,
            sym_delta: d_val,
        }

        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        try:
            gate_allowed = list(dict.fromkeys(["x", "y", "α", "β", "γ", "ε", "μ", "δ"]))
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
            f = f_raw.subs(subs_dict)
            g = g_raw.subs(subs_dict)
        except Exception as e:
            raise ValueError(
                f"Error de sintaxis matemática: revise los símbolos insertados. ({str(e)})"
            )

        f_num = sp.lambdify((x, y), f, "numpy")
        g_num = sp.lambdify((x, y), g, "numpy")

        dx_dt_0 = float(f_num(x0, y0))
        dy_dt_0 = float(g_num(x0, y0))

        is_classic = False
        C_val = 0.0
        winner_analytic = "Indeterminado"
        survivors_analytic = 0.0
        state_eq_latex = ""
        t_end_analytic = -1.0

        # 1. Análisis Teórico de Lanchester
        if (
            sp.simplify(f - (-sym_alpha.subs(subs_dict) * y)) == 0
            and sp.simplify(g - (-sym_beta.subs(subs_dict) * x)) == 0
        ):
            is_classic = True
            C_val = a_val * (y0**2) - b_val * (x0**2)
            state_eq_latex = f"{a_val} y^2 - {b_val} x^2 = {C_val:.2f}"

            try:
                if C_val > 0:
                    winner_analytic = "Ejército Y (Azul)"
                    survivors_analytic = math.sqrt(C_val / a_val) if a_val != 0 else 0
                    if a_val > 0 and b_val > 0:
                        arg = (x0 / y0) * math.sqrt(b_val / a_val)
                        t_end_analytic = (1.0 / math.sqrt(a_val * b_val)) * math.atanh(
                            arg
                        )
                elif C_val < 0:
                    winner_analytic = "Ejército X (Rojo)"
                    survivors_analytic = math.sqrt(-C_val / b_val) if b_val != 0 else 0
                    if a_val > 0 and b_val > 0:
                        arg = (y0 / x0) * math.sqrt(a_val / b_val)
                        t_end_analytic = (1.0 / math.sqrt(a_val * b_val)) * math.atanh(
                            arg
                        )
                else:
                    winner_analytic = "Empate (Aniquilación Mutua)"
            except:
                t_end_analytic = -1.0

        # 2. Simulación RK4 Principal (hasta t_fin)
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
                return np.array([new_x, new_y])
            except:
                return None

        n_steps_t = max(1, int(round((t_fin - t0) / h)))
        t_vals = np.linspace(t0, t_fin, n_steps_t + 1)
        X_main = [np.array([x0, y0])]
        t_actual = [t0]

        winner_numeric = "Combate en curso"
        for i in range(len(t_vals) - 1):
            curr = X_main[-1]
            if curr is None:
                break
            next_p = rk4_step(curr, h)

            # NUEVO: Se considera aniquilación al tener menos de 1 soldado
            if next_p[0] < 1.0 and next_p[1] < 1.0:
                X_main.append(np.array([0.0, 0.0]))
                t_actual.append(t_vals[i + 1])
                winner_numeric = "Empate (Aniquilación Mutua)"
                break
            elif next_p[0] < 1.0:
                X_main.append(np.array([0.0, next_p[1]]))
                t_actual.append(t_vals[i + 1])
                winner_numeric = "Ejército Y (Azul)"
                break
            elif next_p[1] < 1.0:
                X_main.append(np.array([next_p[0], 0.0]))
                t_actual.append(t_vals[i + 1])
                winner_numeric = "Ejército X (Rojo)"
                break
            else:
                X_main.append(next_p)
                t_actual.append(t_vals[i + 1])

        X_main_final = np.array(X_main)

        # 3. Proyección Extendida en la Sombra
        t_end_numeric_str = ""
        if winner_numeric != "Combate en curso":
            t_end_numeric_str = (
                f"Finalizó dentro de la simulación en t = {t_actual[-1]:.2f}"
            )
        else:
            curr_sim = X_main[-1]
            t_sim = t_actual[-1]
            max_t_sim = t0 + 2000.0

            while t_sim < max_t_sim:
                next_p = rk4_step(curr_sim, h * 2)
                if next_p is None:
                    t_end_numeric_str = "Error numérico proyectando al futuro."
                    break
                t_sim += h * 2

                # NUEVO: Menos de 1 soldado
                if next_p[0] < 1.0 and next_p[1] < 1.0:
                    t_end_numeric_str = f"Empate proyectado en t ≈ {t_sim:.2f}"
                    break
                elif next_p[0] < 1.0:
                    t_end_numeric_str = f"Y (Azul) ganará proyectado en t ≈ {t_sim:.2f}"
                    break
                elif next_p[1] < 1.0:
                    t_end_numeric_str = f"X (Rojo) ganará proyectado en t ≈ {t_sim:.2f}"
                    break

                curr_sim = next_p

            if not t_end_numeric_str:
                t_end_numeric_str = "El modelo decae asintóticamente (nunca llega a 0 tropas o excede el límite temporal)."

        # Generar trayectorias de fase representativas
        automatic_trajectories = []
        x_max_plot = max(x0 * 1.5, 10)
        y_max_plot = max(y0 * 1.5, 10)
        x0_auto = np.linspace(1, x_max_plot, 4)
        y0_auto = np.linspace(1, y_max_plot, 4)
        for xi in x0_auto:
            for yi in y0_auto:
                X_temp = [np.array([xi, yi])]
                for _ in range(int(len(t_vals) / 2)):
                    curr = X_temp[-1]
                    if curr[0] <= 0 or curr[1] <= 0:
                        break
                    next_p = rk4_step(curr, h * 3)
                    if next_p[0] < 0:
                        next_p[0] = 0
                    if next_p[1] < 0:
                        next_p[1] = 0
                    X_temp.append(next_p)
                X_temp_final = np.array(X_temp)
                if len(X_temp_final) > 1:
                    automatic_trajectories.append(
                        {
                            "x": X_temp_final[:, 0].tolist(),
                            "y": X_temp_final[:, 1].tolist(),
                        }
                    )

        return {
            "ecuacion_latex_x": sp.latex(f_raw),
            "ecuacion_latex_y": sp.latex(g_raw),
            "dx_dt_0": dx_dt_0,
            "dy_dt_0": dy_dt_0,
            "is_classic": is_classic,
            "C_val": C_val,
            "state_eq_latex": state_eq_latex,
            "winner_analytic": winner_analytic,
            "survivors_analytic": survivors_analytic,
            "t_end_analytic": t_end_analytic,
            "t_end_numeric_str": t_end_numeric_str,
            "winner_numeric": winner_numeric,
            "final_x": X_main_final[-1][0],
            "final_y": X_main_final[-1][1],
            "principal_trajectory": {
                "t": t_actual,
                "x": X_main_final[:, 0].tolist(),
                "y": X_main_final[:, 1].tolist(),
            },
            "automatic_trajectories": automatic_trajectories,
        }
