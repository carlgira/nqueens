"""
Metodo de Prony sobre las 40 soluciones de las 7 reinas.

Cada reina es un complejo  z_j = r[j]·e^{i·a·sol[j]}. Prony halla el numero
MINIMO de modos geometricos p tal que  z_j = sum_k A_k w_k^j , resolviendo la
recurrencia lineal  z_j = c_1 z_{j-1} + ... + c_p z_{j-p}  y midiendo el residuo.
Los modos w_k son las raices del polinomio caracteristico.

Prediccion:  espirales (grado 1) -> p=1 (un solo w).  Feas -> p mayor.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from spiral_queens import SpiralQueens
from discover_queens_formula import solve_queens, min_degree_poly, fmt_poly


def complex_points(sq, sol):
    theta = np.array([sq.a[v] for v in sol])
    return sq.r * np.exp(1j * theta)


def prony_min_p(z, tol=1e-6, maxp=3):
    """Menor p con recurrencia lineal exacta. Devuelve (p, coeffs c, modos w)."""
    N = len(z)
    scale = np.max(np.abs(z))
    for p in range(1, maxp + 1):
        if N - p < p:            # necesitamos al menos p ecuaciones (sobredet.)
            break
        A = np.array([[z[i - 1 - k] for k in range(p)] for i in range(p, N)])
        b = np.array([z[i] for i in range(p, N)])
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = np.max(np.abs(A @ c - b)) / scale
        if resid < tol:
            w = np.roots(np.concatenate(([1.0], -c)))   # raices del pol. caracteristico
            return p, c, w
    return maxp + 1, None, None


def main():
    n = 7
    sq = SpiralQueens(n)
    sols = solve_queens(n)

    rows = []
    for s in sols:
        d, cpoly = min_degree_poly(s)
        z = complex_points(sq, s)
        p, c, w = prony_min_p(z)
        rows.append((s, d, cpoly, p, w))

    # --- distribucion p vs grado ---
    print("Relacion grado (GF7)  <->  p (nº minimo de modos de Prony):")
    print("  grado | p | cuantas")
    from collections import Counter
    combos = Counter((d, p) for _, d, _, p, _ in rows)
    for (d, p), cnt in sorted(combos.items()):
        print(f"    {d}   | {p} |  {cnt}")
    print(f"\n  total: {len(rows)} soluciones")

    # --- ejemplos de modos por cada grado ---
    print("\nEjemplos (modos w_k, |w| = crecimiento, arg = giro):")
    seen = set()
    for s, d, cpoly, p, w in rows:
        if d in seen:
            continue
        seen.add(d)
        print(f"  grado {d}, p={p}:  sol={s}")
        print(f"     {fmt_poly(cpoly)}")
        for wk in (w if w is not None else []):
            print(f"     w = {wk.real:+.3f}{wk.imag:+.3f}i   |w|={abs(wk):.3f}  arg={math.degrees(np.angle(wk)):+.1f}°")

    # --- sanidad: ¿el p=4 de las feas es real? residuo minimo a p=3 ---
    print("\nSanidad (residuo relativo del mejor ajuste a p=3, feas deberian ser >> 0):")
    for s, d, cpoly, p, w in rows:
        if d == 1:
            continue
        z = complex_points(sq, s)
        N = len(z)
        A = np.array([[z[i - 1 - k] for k in range(3)] for i in range(3, N)])
        b = np.array([z[i] for i in range(3, N)])
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        r = np.max(np.abs(A @ c - b)) / np.max(np.abs(z))
        print(f"  grado {d}  sol={s}  residuo(p=3)={r:.3f}")
        if d == 5:
            pass

    # --- paso entre reinas: espiral usa UN paso; fea usa varios ---
    # arg(z_{j+1}/z_j) = (2π/7)·(sol[j+1]-sol[j])  -> el "paso" mod 7.
    def steps(sol):
        return [(sol[j + 1] - sol[j]) % n for j in range(n - 1)]

    print("\nPasos consecutivos (sol[j+1]-sol[j]) mod 7:")
    print("  espiral (grado 1): un solo paso repetido. fea: varios pasos, con ±1 prohibido")
    ugly_with_forbidden = sum(1 for s, d, _, _, _ in rows if d > 1 and (1 in steps(s) or 6 in steps(s)))
    print(f"  feas que usan un paso prohibido (±1): {ugly_with_forbidden}/12")

    # --- figura: p (nº de modos de Prony) de las 40, ordenadas ---
    fig, ax = plt.subplots(figsize=(11, 4))
    order = sorted(range(len(rows)), key=lambda k: (rows[k][3], rows[k][1]))
    color = {1: "#1f77b4", 4: "#ff7f0e", 5: "#d62728"}
    xs = range(len(order))
    ax.bar(xs, [rows[k][3] for k in order],
           color=[color[rows[k][1]] for k in order])
    ax.set_xlabel("las 40 soluciones (ordenadas)")
    ax.set_ylabel("p = nº mínimo de modos de Prony")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_title("Prony: 28 soluciones = 1 espiral (p=1, azul) · 12 = irreducibles (p=4)\n"
                 "naranja grado 4, rojo grado 5 — no hay término medio")
    fig.tight_layout(); fig.savefig("prony_modes.png", dpi=130)
    print("\nGuardado: prony_modes.png")


if __name__ == "__main__":
    main()
