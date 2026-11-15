"""
REMATE: ¿la topología (enlace / clase de homología en el toro de Clifford) refleja
la complejidad algebraica (grado GF7 / modos de Prony) de las 40 soluciones?

Por solución calculamos:
  - grado del polinomio mod 7
  - p de Prony (nº de espirales)
  - w = número de vueltas meridiano del lazo en el toro (clase de homología (1,w))
El enlace de dos soluciones anidadas = w de la exterior, así que w es el invariante
topológico clave. Cruzamos p × w y dibujamos la matriz de enlace completa.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from spiral_queens import SpiralQueens
from prony_sweep import solve_queens, complex_points, prony_min_p
from discover_queens_formula import min_degree_poly

S2 = np.sqrt(2.0)
N = 7


def winding(sol):
    tot = 0
    for r in range(N):
        dv = sol[(r + 1) % N] - sol[r]
        dv = (dv + N // 2) % N - N // 2          # wrap mas corto -> {-3..3}
        tot += dv
    return int(round(tot / N))


def clifford(theta, u, v):
    a = np.cos(theta / 2) * np.cos(u); b = np.cos(theta / 2) * np.sin(u)
    c = np.sin(theta / 2) * np.cos(v); d = np.sin(theta / 2) * np.sin(v)
    k = 1.0 - d
    return np.stack([a / k, b / k, c / k], axis=-1)


def loop(sol, theta, nper=40):
    U, V = [], []
    for r in range(N):
        u0, u1 = 2 * np.pi * r / N, 2 * np.pi * (r + 1) / N
        v0 = 2 * np.pi * sol[r] / N
        dv = ((sol[(r + 1) % N] - sol[r] + N // 2) % N - N // 2)
        v1 = v0 + 2 * np.pi * dv / N
        t = np.linspace(0, 1, nper, endpoint=False)
        U.append(u0 + (u1 - u0) * t); V.append(v0 + (v1 - v0) * t)
    return clifford(theta, np.concatenate(U), np.concatenate(V))


def gauss_link(C1, C2):
    dr1 = np.roll(C1, -1, 0) - C1; dr2 = np.roll(C2, -1, 0) - C2
    m1 = C1 + dr1 / 2; m2 = C2 + dr2 / 2
    R = m1[:, None, :] - m2[None, :, :]
    cross = np.cross(dr1[:, None, :], dr2[None, :, :])
    return np.sum(np.sum(R * cross, -1) / (np.linalg.norm(R, axis=-1) ** 3 + 1e-12)) / (4 * np.pi)


def main():
    sq = SpiralQueens(N)
    sols = solve_queens(N)
    rows = []
    for s in sols:
        d, _ = min_degree_poly(s)
        p = prony_min_p(complex_points(sq, s))
        rows.append((s, d, p, winding(s)))

    # cruce p x w
    print("¿La topología (w) refleja la complejidad (p)?\n")
    print("Cruce  p (Prony) × w (nº de vueltas del lazo):")
    ws = sorted({w for *_, w in rows})
    print("  p \\ w |" + "".join(f"{w:4d}" for w in ws))
    print("  ------+" + "----" * len(ws))
    for p in sorted({r[2] for r in rows}):
        c = Counter(w for _, d, pp, w in rows if pp == p)
        print(f"   {p:2d}   |" + "".join(f"{c.get(w,0):4d}" for w in ws))

    print("\nPor grupo de Prony:")
    for p in sorted({r[2] for r in rows}):
        wl = sorted(w for _, d, pp, w in rows if pp == p)
        print(f"  p={p}: windings = {wl}")

    # matriz de enlace (anidando cada solucion en un theta distinto), ordenada por (p,w)
    order = sorted(range(len(rows)), key=lambda i: (rows[i][2], rows[i][3]))
    thetas = np.linspace(np.pi / 2 - 0.5, np.pi / 2 + 0.5, len(rows))
    curves = [loop(rows[i][0], thetas[k], nper=60) for k, i in enumerate(order)]
    n = len(curves)
    L = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            lk = round(gauss_link(curves[a], curves[b]))
            L[a, b] = L[b, a] = lk

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(L, cmap="coolwarm", vmin=-4, vmax=4)
    # lineas separando grupos de p
    ps = [rows[i][2] for i in order]
    bounds = [k for k in range(1, n) if ps[k] != ps[k - 1]]
    for bnd in bounds:
        ax.axhline(bnd - 0.5, color="k", lw=1); ax.axvline(bnd - 0.5, color="k", lw=1)
    ax.set_title("Matriz de enlace de las 40 soluciones (ordenadas por p de Prony, luego w)\n"
                 "bloques negros = clases de Prony · color = número de enlace")
    fig.colorbar(im, label="número de enlace")
    fig.tight_layout(); fig.savefig("synthesis.png", dpi=130)
    print("\nGuardado: synthesis.png")


if __name__ == "__main__":
    main()
