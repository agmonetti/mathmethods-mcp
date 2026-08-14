import numpy as np
import sympy as sp
import math
from fractions import Fraction
from typing import Dict, Any, List
from .utils import safe_sympify

class Dynamic2DNonHomogeneousService:
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
        n_steps_t = max(1, int(round((t_fin - t0) / h)))
        t_vals = np.linspace(t0, t_fin, n_steps_t + 1)
        X_vals = np.zeros((len(t_vals), 2))
        X_vals[0] = X0

        if callable(B):
            def eval_B(ti):
                return np.array(B(ti), dtype=float)
        else:
            B_vec = np.array(B, dtype=float)

            def eval_B(ti):
                return B_vec

        for i in range(len(t_vals) - 1):
            ti = t_vals[i]
            Xi = X_vals[i]
            h_loc = t_vals[i + 1] - ti

            k1 = A @ Xi + eval_B(ti)
            k2 = A @ (Xi + h_loc * k1 / 2) + eval_B(ti + h_loc / 2)
            k3 = A @ (Xi + h_loc * k2 / 2) + eval_B(ti + h_loc / 2)
            k4 = A @ (Xi + h_loc * k3) + eval_B(ti + h_loc)

            X_vals[i + 1] = Xi + (h_loc / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return t_vals, X_vals

    @staticmethod
    def solve(payload: Dict[str, Any]) -> Dict[str, Any]:
        a = float(payload.get('a', 0.0))
        b = float(payload.get('b', -1.0))
        c = float(payload.get('c', -9.0))
        d = float(payload.get('d', 0.0))
        e_raw = payload.get('e', 1.0)
        f_raw = payload.get('f', 9.0)
        x0 = float(payload.get('x0', 1.0))
        y0 = float(payload.get('y0', 1.0))
        t0 = float(payload.get('t0', 0.0))
        t_fin = float(payload.get('t_fin', 5.0))
        h = float(payload.get('h', 0.01))
        x_min = float(payload.get('x_min', -5.0))
        x_max = float(payload.get('x_max', 5.0))
        y_min = float(payload.get('y_min', -5.0))
        y_max = float(payload.get('y_max', 5.0))
        cantidad_trayectorias = int(payload.get('cantidad_trayectorias', 16))

        t_sym = sp.symbols('t', real=True)

        def parse_forcing_component(value):
            if isinstance(value, (int, float, np.integer, np.floating)):
                return sp.Float(value)
            text = str(value).strip()
            try:
                return safe_sympify(text, local_dict={'t': t_sym, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp, 'pi': sp.pi}, allowed_symbols=['t'])
            except Exception:
                return sp.Float(text)

        e_expr = parse_forcing_component(e_raw)
        f_expr = parse_forcing_component(f_raw)
        forcing_is_time_varying = bool((getattr(e_expr, 'free_symbols', set()) and t_sym in e_expr.free_symbols) or (getattr(f_expr, 'free_symbols', set()) and t_sym in f_expr.free_symbols))

        A = np.array([[a, b], [c, d]], dtype=float)
        if forcing_is_time_varying:
            e_func = sp.lambdify(t_sym, e_expr, 'numpy')
            f_func = sp.lambdify(t_sym, f_expr, 'numpy')

            def B(ti):
                return np.array([float(e_func(ti)), float(f_func(ti))], dtype=float)
        else:
            B = np.array([float(sp.N(e_expr)), float(sp.N(f_expr))], dtype=float)

        det_A = float(np.linalg.det(A))
        equilibrio_unico = abs(det_A) > 1e-12 and not forcing_is_time_varying
        equilibrio_pnto = None
        if equilibrio_unico:
            eq = -np.linalg.solve(A, B)
            equilibrio_pnto = {"x": float(eq[0]), "y": float(eq[1])}

        autovalores, autovectores = np.linalg.eig(A)
        # Fijar un orden estable: primero el autovalor con parte real mayor.
        # Esto hace que v1/v2 no cambien de lugar entre ejecuciones.
        order = np.argsort([-float(np.real(av)) for av in autovalores])
        autovalores = autovalores[order]
        autovectores = autovectores[:, order]
        traza = float(np.trace(A))
        discriminante = traza**2 - 4 * det_A
        clasificacion = Dynamic2DNonHomogeneousService.clasificar_sistema(A)
        comportamiento = Dynamic2DNonHomogeneousService.descripcion_comportamiento(clasificacion)

        list_autovalores = []
        for av in autovalores:
            list_autovalores.append({
                "real": float(np.real(av)),
                "imag": float(np.imag(av))
            })

        list_autovectores = []
        for i in range(len(autovalores)):
            list_autovectores.append({
                "vx": float(np.real(autovectores[0, i])),
                "vy": float(np.real(autovectores[1, i]))
            })

        X0 = np.array([x0, y0], dtype=float)
        t_vals, X_vals = Dynamic2DNonHomogeneousService.rk4_sistema(A, B, X0, t0, t_fin, h)
        
        principal_trajectory = {
            "t": t_vals.tolist(),
            "x": X_vals[:, 0].tolist(),
            "y": X_vals[:, 1].tolist(),
        }

        automatic_trajectories = []
        n_side = int(math.ceil(math.sqrt(cantidad_trayectorias)))
        if n_side > 1:
            x0_auto = np.linspace(x_min, x_max, n_side)
            y0_auto = np.linspace(y_min, y_max, n_side)
            for xi in x0_auto:
                for yi in y0_auto:
                    _, X_auto = Dynamic2DNonHomogeneousService.rk4_sistema(A, B, np.array([xi, yi]), t0, t_fin, h)
                    automatic_trajectories.append({
                        "x": X_auto[::5, 0].tolist(),
                        "y": X_auto[::5, 1].tolist()
                    })

        C1, C2 = sp.symbols('C1 C2')
        # Definiciones de funciones simbólicas para generar ecuaciones en LaTeX
        xs, ys = sp.Function('x'), sp.Function('y')

        # 1. Solución Homogénea (forma general) con C1, C2 simbólicos
        # Usamos SymPy para obtener autovalores/autovectores simbólicos más simples
        A_sym = sp.Matrix([[a, b], [c, d]]).applyfunc(sp.nsimplify)
        evects = A_sym.eigenvects()

        # Manejo sencillo: si aparecen autovalores complejos, devolvemos placeholder
        complex_eigen = any([ev[0].has(sp.I) for ev in evects])
        if complex_eigen:
            lam = None
            v = None
            for ev_sym, mult, vecs in evects:
                if sp.im(ev_sym) > 0:
                    lam = ev_sym
                    v = vecs[0]
                    break
            if lam is None:
                lam = evects[0][0]
                v = evects[0][2][0]

            alpha = sp.re(lam)
            beta = sp.im(lam)

            vec = sp.Matrix(v)
            dens = [el.as_numer_denom()[1] for el in vec]
            l = 1
            for den_val in dens:
                l = sp.ilcm(l, den_val)
            vec_int = (vec * l).applyfunc(sp.simplify)

            a_vec = vec_int.applyfunc(sp.re)
            b_vec = vec_int.applyfunc(sp.im)

            nums = []
            for val in list(a_vec) + list(b_vec):
                if val != 0:
                    nums.append(abs(int(val)))
            if nums:
                g = nums[0]
                for n in nums[1:]:
                    g = math.gcd(g, n)
                if g > 1:
                    vec_int = (vec_int / g).applyfunc(sp.simplify)
                    a_vec = vec_int.applyfunc(sp.re)
                    b_vec = vec_int.applyfunc(sp.im)

            autovectores_normalizados = []
            autovectores_relaciones_latex = []
            autovectores_parametricos_latex = []
            
            exp_at = sp.exp(alpha * t_sym)
            cos_bt = sp.cos(beta * t_sym)
            sin_bt = sp.sin(beta * t_sym)

            term1 = a_vec * cos_bt - b_vec * sin_bt
            term2 = a_vec * sin_bt + b_vec * cos_bt

            X1 = sp.simplify(exp_at * term1)
            X2 = sp.simplify(exp_at * term2)

            sol_homogenea_vectorial = sp.simplify(C1 * X1 + C2 * X2)
            
            term1_latex = sp.latex(term1)
            term2_latex = sp.latex(term2)
            exp_latex = sp.latex(exp_at)
            
            sol_homogenea_vectorial_unmultiplied_latex = f"X_h(t) = C_1 {exp_latex} \\left( {term1_latex} \\right) + C_2 {exp_latex} \\left( {term2_latex} \\right)"
            sol_homogenea_vectorial_latex = f"X_h(t) = {sp.latex(sol_homogenea_vectorial)}"

            xh_t = sp.simplify(sol_homogenea_vectorial[0])
            yh_t = sp.simplify(sol_homogenea_vectorial[1])
            sol_homogenea_componentes_latex = [sp.latex(sp.Eq(xs(t_sym), xh_t)), sp.latex(sp.Eq(ys(t_sym), yh_t))]

            if forcing_is_time_varying:
                sol_particular_latex = ["Solución particular simbólica no soportada para forcings variantes con autovalores complejos."]
                Xp_sym = sp.Matrix([0, 0])
            else:
                Xp = np.zeros(2)
                if equilibrio_pnto:
                    Xp = np.array([equilibrio_pnto['x'], equilibrio_pnto['y']])
                Xp_sym = sp.Matrix(Xp)
                sol_particular_latex = [f"X_p = \\begin{{pmatrix}} {Xp[0]:g} \\\\ {Xp[1]:g} \\end{{pmatrix}}"]

            constantes = {"c1": "C1", "c2": "C2"}

            sol_general_vectorial = sp.simplify(sol_homogenea_vectorial + Xp_sym)
            sol_general_vectorial_unmultiplied_latex = f"X(t) = C_1 {exp_latex} \\left( {term1_latex} \\right) + C_2 {exp_latex} \\left( {term2_latex} \\right) + {sp.latex(Xp_sym)}"
            sol_general_vectorial_latex = f"X(t) = {sp.latex(sol_general_vectorial)}"

            x_t = sp.simplify(sol_general_vectorial[0])
            y_t = sp.simplify(sol_general_vectorial[1])
            sol_general_latex = [sp.latex(sp.Eq(xs(t_sym), x_t)), sp.latex(sp.Eq(ys(t_sym), y_t))]
        else:
            # Extraer autovalores y autovectores simples
            # evects: list of tuples (eigenvalue, multiplicity, [eigenvectors])
            # Queremos alinear el orden simbólico con el orden de autovalores de numpy
            def simplify_sym_vector_to_ints(vec_sym: sp.Matrix):
                vec = sp.Matrix(vec_sym)
                dens = [sp.denom(el) for el in vec]
                l = 1
                for d in dens:
                    l = sp.ilcm(l, d)
                vec_int = (vec * l).applyfunc(sp.simplify)
                # convertir a enteros y reducir por GCD
                nums = [int(sp.sign(el) * abs(sp.Integer(el.as_numer_denom()[0]))) for el in vec_int]
                g = 0
                for n in nums:
                    g = abs(n) if g == 0 else math.gcd(g, abs(n))
                if g > 1:
                    nums = [int(n // g) for n in nums]
                # evitar vector cero
                if all(n == 0 for n in nums):
                    return None
                # Preferir primer componente no negativo para presentación
                if nums[0] < 0:
                    nums = [-n for n in nums]
                return nums

            # Construir lista de autovalores simbólicos en el mismo orden que numpy devuelve autovalores
            eigen_syms_ordered = []
            used_evects = [False] * len(evects)
            for idx_num, av in enumerate(autovalores):
                matched = False
                for j, ev in enumerate(evects):
                    if used_evects[j]:
                        continue
                    try:
                        if abs(float(ev[0]) - float(av)) < 1e-6:
                            eigen_syms_ordered.append(ev[0])
                            used_evects[j] = True
                            matched = True
                            break
                    except Exception:
                        continue
                if not matched:
                    # si no encontramos por cercanía, usar el siguiente no usado
                    for j, ev in enumerate(evects):
                        if not used_evects[j]:
                            eigen_syms_ordered.append(ev[0])
                            used_evects[j] = True
                            break

            # Preferir autovectores simbólicos (SymPy) si dan entradas racionales simples.
            v1_nums = None
            v2_nums = None
            try:
                # intentar extraer autovectores simbólicos de SymPy en el orden de numpy
                sym_vecs_ordered = []
                for lam_sym in eigen_syms_ordered:
                    # buscar el evects que corresponde a lam_sym
                    found = False
                    for ev in evects:
                        try:
                            if abs(float(ev[0]) - float(lam_sym)) < 1e-8:
                                sym_vecs_ordered.append(ev[2][0])
                                found = True
                                break
                        except Exception:
                            continue
                    if not found:
                        sym_vecs_ordered.append(None)

                # simplificar cada vector simbólico si existe
                v_nums_list = []
                for sv in sym_vecs_ordered:
                    if sv is not None:
                        v_try = simplify_sym_vector_to_ints(sv)
                        v_nums_list.append(v_try)
                    else:
                        v_nums_list.append(None)

                # asignar por posición (v1 = primero, v2 = segundo) si disponibles
                if len(v_nums_list) >= 1:
                    v1_nums = v_nums_list[0]
                if len(v_nums_list) >= 2:
                    v2_nums = v_nums_list[1]
            except Exception:
                v1_nums = None
                v2_nums = None

            # Si no obtuvimos vectores simbólicos simples, caer al método numérico
            from fractions import Fraction

            def numeric_to_simple_ints(v: np.ndarray, max_den: int = 12):
                comps = [float(x) for x in v]
                frs = [Fraction(c).limit_denominator(max_den) for c in comps]
                dens = [fr.denominator for fr in frs]
                l = 1
                for d in dens:
                    l = l * d // math.gcd(l, d)
                ints = [fr.numerator * (l // fr.denominator) for fr in frs]
                # reducir por GCD
                g = 0
                for val in ints:
                    g = abs(val) if g == 0 else math.gcd(g, abs(val))
                if g > 1:
                    ints = [int(val // g) for val in ints]
                # evitar el vector cero
                if all(val == 0 for val in ints):
                    if abs(comps[0]) >= abs(comps[1]):
                        ints = [1, 0]
                    else:
                        ints = [0, 1]
                # Preferir primer componente no negativo
                if ints[0] < 0:
                    ints = [-val for val in ints]
                return ints

            try:
                if v1_nums is None:
                    v1_nums = numeric_to_simple_ints(autovectores[:, 0].real)
            except Exception:
                v1_nums = [1, 0]
            try:
                if v2_nums is None:
                    v2_nums = numeric_to_simple_ints(autovectores[:, 1].real) if len(autovalores) > 1 else [0, 1]
            except Exception:
                v2_nums = [0, 1]

            # Mantener la versión simplificada numérica (temporal), se reemplazará
            # por la versión normalizada más abajo para la salida y las soluciones.
            temp_autovectores = [
                {"vx": int(v1_nums[0]), "vy": int(v1_nums[1])},
                {"vx": int(v2_nums[0]), "vy": int(v2_nums[1])}
            ]

            # Generar pasos simbólicos para despejar la relación entre componentes del autovector
            k1_sym, k2_sym = sp.symbols('k_1 k_2')
            autovectores_pasos_latex = []
            # guardamos la relación simbólica y el objeto relacional para derivar luego el vector
            autovectores_relaciones_text_sym = []
            autovectores_relaciones_obj = []
            # usar la lista de autovalores simbólicos ordenada para coincidir con numpy
            eigen_syms = eigen_syms_ordered if 'eigen_syms_ordered' in locals() else []
            for lam in eigen_syms:
                M = A_sym - lam * sp.eye(2)
                eq1 = sp.Eq(M[0, 0] * k1_sym + M[0, 1] * k2_sym, 0)
                eq2 = sp.Eq(M[1, 0] * k1_sym + M[1, 1] * k2_sym, 0)
                basis = M.nullspace()
                basis_vec = basis[0] if basis else None
                if basis_vec is not None:
                    basis_nums = simplify_sym_vector_to_ints(basis_vec)
                    if basis_nums is None:
                        basis_nums = [1, 0]
                else:
                    basis_nums = [1, 0]

                if basis_nums[0] == 0 and basis_nums[1] != 0:
                    rel = sp.Eq(k1_sym, 0)
                elif basis_nums[1] == 0 and basis_nums[0] != 0:
                    rel = sp.Eq(k2_sym, 0)
                elif basis_nums[0] != 0:
                    coeff_rel = sp.Rational(basis_nums[1], basis_nums[0])
                    rel = sp.Eq(k2_sym, sp.simplify(coeff_rel * k1_sym))
                else:
                    rel = sp.Eq(k2_sym, sp.Integer(0))
                autovectores_pasos_latex.append({
                    "sistema_latex": [sp.latex(eq1), sp.latex(eq2)],
                    "relacion_latex": sp.latex(rel)
                })
                # construir una versión en texto plano de la relación, p.ej. 'k2 = -3 k1'
                try:
                    lhs = str(rel.lhs).replace('k_1', 'k1').replace('k_2', 'k2')
                    rhs_expr = rel.rhs
                    # si rhs es multiplicación de coeficiente racional y k1/k2, extraer coef y variable
                    if rhs_expr.is_Mul:
                        coeff = None
                        var = None
                        for arg in rhs_expr.args:
                            if arg.is_Number or arg.is_Rational or arg.is_Integer:
                                coeff = arg
                            else:
                                var = arg
                        # formatear coeficiente
                        if coeff is None:
                            coeff_str = '1'
                        elif isinstance(coeff, sp.Rational):
                            if coeff.q == 1:
                                coeff_str = str(int(coeff.p))
                            else:
                                coeff_str = f"{int(coeff.p)}/{int(coeff.q)}"
                        else:
                            coeff_str = str(float(coeff))
                        var_str = str(var).replace('k_1', 'k1').replace('k_2', 'k2') if var is not None else ''
                        rhs_text = f"{coeff_str} {var_str}".strip()
                    else:
                        rhs_text = str(rhs_expr).replace('k_1', 'k1').replace('k_2', 'k2').replace('*', ' ')
                    rel_text = f"{lhs} = {rhs_text}"
                except Exception:
                    rel_text = str(rel).replace('k_1', 'k1').replace('k_2', 'k2')
                # guardar la versión en texto plano de la relación simbólica
                autovectores_relaciones_text_sym.append(rel_text)
                autovectores_relaciones_obj.append(rel)

            # Normalizar a partir de la relación del sistema: primero se despeja, luego se fija la componente libre.
            autovectores_normalizados = []
            autovectores_relaciones_latex = []
            autovectores_parametricos_latex = []

            def relation_to_vector(rel, fallback_nums):
                def simplificar_coefs(coeff):
                    coeff = sp.nsimplify(coeff)
                    if coeff.is_Rational:
                        return coeff
                    try:
                        return sp.Rational(coeff)
                    except Exception:
                        return coeff

                def vector_from_coeff(lhs_is_k2, coeff):
                    coeff = simplificar_coefs(coeff)
                    if coeff == 0:
                        return [1, 0] if lhs_is_k2 else [0, 1]
                    if isinstance(coeff, sp.Rational):
                        p, q = int(coeff.p), int(coeff.q)
                        if lhs_is_k2:
                            return [q, p]
                        return [p, q]
                    try:
                        val = float(coeff)
                        rounded = int(round(val))
                        return [1, rounded] if lhs_is_k2 else [rounded, 1]
                    except Exception:
                        a, b = fallback_nums[0], fallback_nums[1]
                        return [int(a), int(b)]

                if rel is None:
                    a, b = fallback_nums[0], fallback_nums[1]
                    if a == 0 and b == 0:
                        return [1, 0]
                    if a == 0:
                        return [0, 1]
                    ratio = Fraction(b, a)
                    if ratio.denominator == 1:
                        return [1, int(ratio)]
                    return [ratio.denominator, ratio.numerator]

                if rel.lhs == k2_sym:
                    coeff = sp.simplify(rel.rhs / k1_sym) if rel.rhs != 0 else sp.Integer(0)
                    vec = vector_from_coeff(True, coeff)
                elif rel.lhs == k1_sym:
                    coeff = sp.simplify(rel.rhs / k2_sym) if rel.rhs != 0 else sp.Integer(0)
                    vec = vector_from_coeff(False, coeff)
                else:
                    a, b = fallback_nums[0], fallback_nums[1]
                    vec = [int(a), int(b)]

                if vec[0] == 0 and vec[1] == 0:
                    vec = [1, 0]
                if vec[0] != 0:
                    gcd = math.gcd(abs(int(vec[0])), abs(int(vec[1]))) if vec[1] != 0 else abs(int(vec[0]))
                    if gcd > 1:
                        vec = [int(vec[0] // gcd), int(vec[1] // gcd)]
                return vec

            def relation_text_from_vector(vec):
                vx, vy = int(vec[0]), int(vec[1])
                if vx == 0 and vy == 0:
                    return ""
                if vy == 0:
                    return "k2 = 0"
                if vx == 0:
                    return "k1 = 0"
                ratio = Fraction(vy, vx)
                if ratio.denominator == 1:
                    return f"k2 = {ratio.numerator} k1"
                return f"k2 = {ratio.numerator}/{ratio.denominator} k1"

            rel_v1 = autovectores_relaciones_obj[0] if len(autovectores_relaciones_obj) > 0 else None
            rel_v2 = autovectores_relaciones_obj[1] if len(autovectores_relaciones_obj) > 1 else None

            v1_norm = relation_to_vector(rel_v1, v1_nums)
            v2_norm = relation_to_vector(rel_v2, v2_nums)
            v1_rel = relation_text_from_vector(v1_norm)
            v2_rel = relation_text_from_vector(v2_norm)
            v1_param = f"\\mathbf{{v}} = k \\begin{{pmatrix}}{v1_norm[0]}\\\\{v1_norm[1]}\\end{{pmatrix}}"
            v2_param = f"\\mathbf{{v}} = k \\begin{{pmatrix}}{v2_norm[0]}\\\\{v2_norm[1]}\\end{{pmatrix}}"

            autovectores_normalizados.append({"vx": v1_norm[0], "vy": v1_norm[1]})
            autovectores_normalizados.append({"vx": v2_norm[0], "vy": v2_norm[1]})
            autovectores_relaciones_latex.append(v1_rel)
            autovectores_relaciones_latex.append(v2_rel)
            autovectores_parametricos_latex.append(v1_param)
            autovectores_parametricos_latex.append(v2_param)

            # La relación textual se deriva del autovector normalizado.
            autovectores_relaciones_text = [v1_rel, v2_rel]

            # Usar los autovectores *normalizados* para construir las expresiones simbólicas
            v1_sym = sp.Matrix([autovectores_normalizados[0]["vx"], autovectores_normalizados[0]["vy"]])
            v2_sym = sp.Matrix([autovectores_normalizados[1]["vx"], autovectores_normalizados[1]["vy"]])

            # Actualizar la lista de autovectores de salida a las versiones normalizadas
            list_autovectores = autovectores_normalizados

            # Asegurar que lambda1_sym/lambda2_sym estén definidos (usar el orden simbólico si existe)
            if 'eigen_syms_ordered' in locals() and len(eigen_syms_ordered) > 0:
                lambda1_sym = eigen_syms_ordered[0]
                lambda2_sym = eigen_syms_ordered[1] if len(eigen_syms_ordered) > 1 else eigen_syms_ordered[0]
            else:
                # fallback: usar primeros autovalores simbólicos de evects
                lambda1_sym = evects[0][0]
                lambda2_sym = evects[1][0] if len(evects) > 1 else evects[0][0]

            # Solución homogénea: forma con autovectores sin multiplicar (C1 e^{λ1 t} v1 + C2 e^{λ2 t} v2)
            scalar1 = sp.simplify(C1 * sp.exp(lambda1_sym * t_sym))
            scalar2 = sp.simplify(C2 * sp.exp(lambda2_sym * t_sym))
            vec1_latex = sp.latex(v1_sym)
            vec2_latex = sp.latex(v2_sym)
            sol_homogenea_vectorial_unmultiplied_latex = f"X_h(t) = {sp.latex(scalar1)} {vec1_latex} + {sp.latex(scalar2)} {vec2_latex}"

            # Versión multiplicada y simplificada (componentes explícitas)
            sol_homogenea_vectorial = C1 * v1_sym * sp.exp(lambda1_sym * t_sym) + C2 * v2_sym * sp.exp(lambda2_sym * t_sym)
            sol_homogenea_vectorial_latex = f"X_h(t) = {sp.latex(sol_homogenea_vectorial)}"

            # Componentes x_h(t) y y_h(t) de la solución homogénea (multiplicada)
            xh_t = sp.simplify(sol_homogenea_vectorial[0])
            yh_t = sp.simplify(sol_homogenea_vectorial[1])
            sol_homogenea_componentes_latex = [sp.latex(sp.Eq(xs(t_sym), xh_t)), sp.latex(sp.Eq(ys(t_sym), yh_t))]

            # 2. Solución Particular
            if forcing_is_time_varying and abs(b) < 1e-12 and abs(c) < 1e-12:
                # Caso diagonal: resolver componente por componente con factor integrante.
                def trig_particular(forcing_expr, coeff):
                    # Solución particular de  y' - coeff*y = forcing(t)  por factor
                    # integrante:  y_p(t) = e^{coeff t} ∫ e^{-coeff t} forcing(t) dt.
                    # Vale para CUALQUIER forzado, incluidas frecuencias ω ≠ 1
                    # (antes, el ensayo u·cos(t)+v·sin(t) solo era correcto para ω=1
                    #  y devolvía soluciones que no satisfacían la EDO para ω≠1).
                    raw = sp.simplify(
                        sp.exp(coeff * t_sym)
                        * sp.integrate(sp.exp(-coeff * t_sym) * forcing_expr, t_sym)
                    )
                    return sp.expand_trig(raw)

                x_p_t = trig_particular(e_expr, a)
                y_p_t = trig_particular(f_expr, d)
                Xp_sym = sp.Matrix([x_p_t, y_p_t])
                sol_particular_latex = [f"X_p(t) = {sp.latex(Xp_sym)}"]
                Xp = np.array([0.0, 0.0], dtype=float)
            else:
                Xp = np.zeros(2)
                if equilibrio_pnto:
                    Xp = np.array([equilibrio_pnto['x'], equilibrio_pnto['y']])
                # Mostrar la solución particular redondeada a 0 decimales para presentación
                sol_particular_latex = [f"X_p = \\begin{{pmatrix}} {Xp[0]:g} \\\\ {Xp[1]:g} \\end{{pmatrix}}"]

            # 3. No calculamos numéricamente C1, C2: los dejamos simbólicos
            constantes = {"c1": "C1", "c2": "C2"}

            # 4. Solución general vectorial simbólica: mostrar primero la parte homogénea sin multiplicar
            if forcing_is_time_varying and abs(b) < 1e-12 and abs(c) < 1e-12:
                sol_general_vectorial_unmultiplied_latex = f"X(t) = {sp.latex(scalar1)} {vec1_latex} + {sp.latex(scalar2)} {vec2_latex} + {sp.latex(Xp_sym)}"
            else:
                Xp_sym = sp.Matrix(Xp)
                sol_general_vectorial_unmultiplied_latex = f"X(t) = {sp.latex(scalar1)} {vec1_latex} + {sp.latex(scalar2)} {vec2_latex} + {sp.latex(Xp_sym)}"

            # 5. Solución general multiplicada y simplificada (componentes explícitas)
            if forcing_is_time_varying and abs(b) < 1e-12 and abs(c) < 1e-12:
                sol_general_vectorial = sol_homogenea_vectorial + Xp_sym
            else:
                sol_general_vectorial = sol_homogenea_vectorial + Xp_sym
            sol_general_vectorial_latex = f"X(t) = {sp.latex(sol_general_vectorial)}"

            x_t = sp.simplify(sol_general_vectorial[0])
            y_t = sp.simplify(sol_general_vectorial[1])
            sol_general_latex = [sp.latex(sp.Eq(xs(t_sym), x_t)), sp.latex(sp.Eq(ys(t_sym), y_t))]

        # Si definimos autovectores_pasos_latex en el flujo anterior, úsalo; si no, vacío
        try:
            autovectores_pasos_latex
        except NameError:
            autovectores_pasos_latex = []
        try:
            autovectores_relaciones_text
        except NameError:
            autovectores_relaciones_text = []

        real_eigenvectors_lines = []
        if equilibrio_unico and equilibrio_pnto:
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
            "vector_b": [e_raw, f_raw],
            "traza": traza,
            "determinante": det_A,
            "clasificacion_homogenea": clasificacion,
            "comportamiento": comportamiento,
            "equilibrio": equilibrio_pnto,
            "autovalores": list_autovalores,
            "autovectores": list_autovectores,
            "autovectores_normalizados": autovectores_normalizados,
            "autovectores_relaciones_latex": autovectores_relaciones_latex,
            "autovectores_parametricos_latex": autovectores_parametricos_latex,
            "autovectores_pasos_latex": autovectores_pasos_latex,
            "autovectores_relaciones_text": autovectores_relaciones_text,
            "constantes": constantes,
            "solucion_homogenea_vectorial_latex": sol_homogenea_vectorial_latex,
            "solucion_homogenea_vectorial_unmultiplied_latex": sol_homogenea_vectorial_unmultiplied_latex,
            "solucion_homogenea_componentes_latex": sol_homogenea_componentes_latex,
            "solucion_particular_latex": sol_particular_latex,
            "solucion_general_vectorial_unmultiplied_latex": sol_general_vectorial_unmultiplied_latex,
            "solucion_general_vectorial_latex": sol_general_vectorial_latex,
            "solucion_general_latex": sol_general_latex,
            "principal_trajectory": principal_trajectory,
            "automatic_trajectories": automatic_trajectories,
            "real_eigenvectors_lines": real_eigenvectors_lines
        }

