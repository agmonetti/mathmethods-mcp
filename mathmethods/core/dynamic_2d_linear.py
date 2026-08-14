import numpy as np
import sympy as sp
import math
from typing import Dict, Any, List

class Dynamic2DLinearService:
    @staticmethod
    def clasificar_sistema(A) -> str:
        traza = float(np.trace(A))
        determinante = float(np.linalg.det(A))
        discriminante = traza**2 - 4 * determinante
        autovalores = np.linalg.eigvals(A)

        if abs(determinante) < 1e-10:
            return "Caso degenerado / determinante cero"
        if determinante < 0:
            return "Silla"
        if discriminante > 1e-10:
            if traza < 0:
                return "Nodo estable"
            elif traza > 0:
                return "Nodo inestable"
            else:
                return "Nodo con traza cero / caso especial"
        elif abs(discriminante) <= 1e-10:
            if traza < 0:
                return "Nodo degenerado estable"
            elif traza > 0:
                return "Nodo degenerado inestable"
            else:
                return "Caso no hiperbolico"
        else:
            parte_real = float(np.real(autovalores[0]))
            if abs(parte_real) < 1e-10:
                return "Centro"
            elif parte_real < 0:
                return "Foco estable"
            else:
                return "Foco inestable"

    @staticmethod
    def descripcion_comportamiento(clasificacion: str) -> str:
        if "Silla" in clasificacion:
            return "Las trayectorias se acercan en una direccion y se alejan en otra."
        if "Nodo estable" in clasificacion:
            return "Todas las trayectorias cercanas tienden al equilibrio sin oscilar."
        if "Nodo inestable" in clasificacion:
            return "Las trayectorias se alejan del equilibrio sin oscilar."
        if "Foco estable" in clasificacion:
            return "Las trayectorias giran en espiral hacia el equilibrio."
        if "Foco inestable" in clasificacion:
            return "Las trayectorias giran en espiral alejandose del equilibrio."
        if "Centro" in clasificacion:
            return "Las trayectorias cerradas rodean el equilibrio."
        if "degenerado" in clasificacion:
            return "El sistema tiene autovalores repetidos; puede requerir analizar autovectores."
        return "Caso especial; requiere analisis adicional."

    @staticmethod
    def rk4_sistema(A, B, X0, t0, t_fin, h):
        t_vals = np.arange(t0, t_fin + h, h)
        X_vals = np.zeros((len(t_vals), 2))
        X_vals[0] = X0

        for i in range(len(t_vals) - 1):
            ti = t_vals[i]
            Xi = X_vals[i]
            h_loc = t_vals[i + 1] - ti

            k1 = A @ Xi + B
            k2 = A @ (Xi + h_loc * k1 / 2) + B
            k3 = A @ (Xi + h_loc * k2 / 2) + B
            k4 = A @ (Xi + h_loc * k3) + B

            X_vals[i + 1] = Xi + (h_loc / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return t_vals, X_vals

    @staticmethod
    def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
        a = float(payload.get('a', 3.0))
        b = float(payload.get('b', 1.0))
        c = float(payload.get('c', 1.0))
        d = float(payload.get('d', 3.0))
        e = float(payload.get('e', 0.0))
        f = float(payload.get('f', 0.0))
        x0 = float(payload.get('x0', 1.0))
        y0 = float(payload.get('y0', 1.0))
        t0 = float(payload.get('t0', 0.0))
        t_fin = float(payload.get('t_fin', 10.0))
        h = float(payload.get('h', 0.01))
        x_min = float(payload.get('x_min', -5.0))
        x_max = float(payload.get('x_max', 5.0))
        y_min = float(payload.get('y_min', -5.0))
        y_max = float(payload.get('y_max', 5.0))
        cantidad_trayectorias = int(payload.get('cantidad_trayectorias', 16))

        A = np.array([[a, b], [c, d]], dtype=float)
        B = np.array([e, f], dtype=float)

        det_A = float(np.linalg.det(A))
        equilibrio_unico = abs(det_A) > 1e-12
        equilibrio_pnto = None
        if equilibrio_unico:
            eq = -np.linalg.solve(A, B)
            equilibrio_pnto = {"x": float(eq[0]), "y": float(eq[1])}

        autovalores, autovectores = np.linalg.eig(A)
        traza = float(np.trace(A))
        discriminante = traza**2 - 4 * det_A
        clasificacion = Dynamic2DLinearService.clasificar_sistema(A)
        comportamiento = Dynamic2DLinearService.descripcion_comportamiento(clasificacion)

        list_autovalores = []
        for av in autovalores:
            list_autovalores.append({
                "real": float(np.real(av)),
                "imag": float(np.imag(av))
            })

        list_autovectores = []
        for i in range(len(autovalores)):
            list_autovectores.append({
                "vx": float(autovectores[0, i]),
                "vy": float(autovectores[1, i])
            })

        X0 = np.array([x0, y0], dtype=float)
        t_vals, X_vals = Dynamic2DLinearService.rk4_sistema(A, B, X0, t0, t_fin, h)
        
        dxdt_vals = [float(val) for val in (A @ X_vals.T + B[:, None])[0]]
        dydt_vals = [float(val) for val in (A @ X_vals.T + B[:, None])[1]]

        principal_trajectory = {
            "t": t_vals.tolist(),
            "x": X_vals[:, 0].tolist(),
            "y": X_vals[:, 1].tolist(),
            "dxdt": dxdt_vals,
            "dydt": dydt_vals
        }

        automatic_trajectories = []
        n_side = int(math.sqrt(cantidad_trayectorias))
        if n_side > 1:
            x0_auto = np.linspace(x_min, x_max, n_side)
            y0_auto = np.linspace(y_min, y_max, n_side)
            for xi in x0_auto:
                for yi in y0_auto:
                    _, X_auto = Dynamic2DLinearService.rk4_sistema(A, B, np.array([xi, yi]), t0, t_fin, h)
                    automatic_trajectories.append({
                        "x": X_auto[::5, 0].tolist(),
                        "y": X_auto[::5, 1].tolist()
                    })

        # Limpieza de decimales nulos (.0) para SymPy
        def to_num(val):
            return int(val) if val.is_integer() else val

        a_sym, b_sym = to_num(a), to_num(b)
        c_sym, d_sym = to_num(c), to_num(d)
        e_sym, f_sym = to_num(e), to_num(f)
        x0_sym, y0_sym = to_num(x0), to_num(y0)
        t0_sym = to_num(t0)

        xs = sp.Function('x')
        ys = sp.Function('y')
        t_sym = sp.Symbol('t', real=True)

        sol_homogenea_latex = []
        try:
            # Para la solución homogénea, forzamos B a ser [0, 0]
            eq1_h = sp.Eq(sp.diff(xs(t_sym), t_sym), a_sym * xs(t_sym) + b_sym * ys(t_sym))
            eq2_h = sp.Eq(sp.diff(ys(t_sym), t_sym), c_sym * xs(t_sym) + d_sym * ys(t_sym))
            sol_homogenea = sp.dsolve([eq1_h, eq2_h], ics={xs(t0_sym): x0_sym, ys(t0_sym): y0_sym})
            
            if isinstance(sol_homogenea, list):
                sol_homogenea_latex = [sp.latex(eq) for eq in sol_homogenea]
            else:
                sol_homogenea_latex = [sp.latex(sol_homogenea)]
        except Exception:
            sol_homogenea_latex = ["SymPy no pudo resolver la parte homogénea."]

        # Solución General (No Homogénea)
        sol_general_latex = []
        try:
            eq1_g = sp.Eq(sp.diff(xs(t_sym), t_sym), a_sym * xs(t_sym) + b_sym * ys(t_sym) + e_sym)
            eq2_g = sp.Eq(sp.diff(ys(t_sym), t_sym), c_sym * xs(t_sym) + d_sym * ys(t_sym) + f_sym)
            sol_general = sp.dsolve([eq1_g, eq2_g], ics={xs(t0_sym): x0_sym, ys(t0_sym): y0_sym})
            
            if isinstance(sol_general, list):
                sol_general_latex = [sp.latex(eq) for eq in sol_general]
            else:
                sol_general_latex = [sp.latex(sol_general)]
        except Exception:
            sol_general_latex = ["SymPy no pudo resolver el sistema completo."]

        # Solución Particular (Punto de equilibrio si B es constante)
        sol_particular_latex = []
        if equilibrio_unico and equilibrio_pnto:
            sol_particular_latex = [
                f"x_p(t) = {equilibrio_pnto['x']}",
                f"y_p(t) = {equilibrio_pnto['y']}"
            ]

        # Nulclinas
        nulclina_x_pts = []
        if abs(b) > 1e-9:
            nulclina_x_pts = [[x_min, (-a * x_min - e) / b], [x_max, (-a * x_max - e) / b]]
        elif abs(a) > 1e-9:
            nulclina_x_pts = [[-e / a, y_min], [-e / a, y_max]]

        nulclina_y_pts = []
        if abs(d) > 1e-9:
            nulclina_y_pts = [[x_min, (-c * x_min - f) / d], [x_max, (-c * x_max - f) / d]]
        elif abs(c) > 1e-9:
            nulclina_y_pts = [[-f / c, y_min], [-f / c, y_max]]

        # Despejar y(x) en las Nulclinas con SymPy
        x_sym, y_sym_var = sp.symbols('x y', real=True)
        
        nul_x_expr = sp.Eq(a_sym * x_sym + b_sym * y_sym_var + e_sym, 0)
        try:
            sol_x = sp.solve(nul_x_expr, y_sym_var)
            nul_x_despejada = f"y = {str(sol_x[0]).replace('**', '^').replace('*', ' ')}" if sol_x else "No despejable como y(x)"
        except Exception:
            nul_x_despejada = "No despejable como y(x)"

        nul_y_expr = sp.Eq(c_sym * x_sym + d_sym * y_sym_var + f_sym, 0)
        try:
            sol_y = sp.solve(nul_y_expr, y_sym_var)
            nul_y_despejada = f"y = {str(sol_y[0]).replace('**', '^').replace('*', ' ')}" if sol_y else "No despejable como y(x)"
        except Exception:
            nul_y_despejada = "No despejable como y(x)"

        real_eigenvectors_lines = []
        if equilibrio_unico:
            for i in range(len(autovalores)):
                if abs(np.imag(autovalores[i])) < 1e-10:
                    vx, vy = float(np.real(autovectores[0, i])), float(np.real(autovectores[1, i]))
                    norm = math.sqrt(vx**2 + vy**2)
                    if norm > 1e-9:
                        vx, vy = vx / norm, vy / norm
                        real_eigenvectors_lines.append([
                            [equilibrio_pnto["x"] - 20 * vx, equilibrio_pnto["y"] - 20 * vy],
                            [equilibrio_pnto["x"] + 20 * vx, equilibrio_pnto["y"] + 20 * vy]
                        ])

        return {
            "matrix_a": [[a, b], [c, d]],
            "vector_b": [e, f],
            "traza": traza,
            "determinante": det_A,
            "discriminante": discriminante,
            "clasificacion": clasificacion,
            "comportamiento": comportamiento,
            "equilibrio": equilibrio_pnto,
            "equilibrio_unico": equilibrio_unico,
            "autovalores": list_autovalores,
            "autovectores": list_autovectores,
            "solucion_homogenea_latex": sol_homogenea_latex,
            "solucion_particular_latex": sol_particular_latex,
            "solucion_general_latex": sol_general_latex,
            "nulclina_x": {"puntos": nulclina_x_pts, "ecuacion_despejada": nul_x_despejada},
            "nulclina_y": {"puntos": nulclina_y_pts, "ecuacion_despejada": nul_y_despejada},
            "principal_trajectory": principal_trajectory,
            "automatic_trajectories": automatic_trajectories,
            "real_eigenvectors_lines": real_eigenvectors_lines
        }