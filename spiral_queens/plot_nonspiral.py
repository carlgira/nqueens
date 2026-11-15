"""
El "camino" de las 40 soluciones de las 7 reinas sobre la rejilla polar.

Para las 28 espirales (grado 1) teniamos una espiral generadora. Para las 12
restantes (grado 4 y 5 sobre GF(7)) no hay espiral, pero SI podemos dibujar el
camino que une las reinas en orden de radio (j=0..6) con una curva suave
(Catmull-Rom). Mismo metodo para todas:
  - grado 1  -> el camino es una espiral limpia
  - grado>=2 -> el camino es un roseton que se cruza ("sin espiral")
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from spiral_queens import SpiralQueens, polar_2_cartesian
from discover_queens_formula import solve_queens, min_degree_poly, fmt_poly


def catmull_rom(P, samples=24):
    """Curva suave abierta que pasa por los puntos P (n,2)."""
    P = np.asarray(P, float)
    Pp = np.vstack([P[0] - (P[1] - P[0]), P, P[-1] + (P[-1] - P[-2])])
    out = []
    for i in range(1, len(Pp) - 2):
        p0, p1, p2, p3 = Pp[i - 1], Pp[i], Pp[i + 1], Pp[i + 2]
        t = np.linspace(0, 1, samples)[:, None]
        seg = 0.5 * (2 * p1 + (-p0 + p2) * t
                     + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
        out.append(seg)
    return np.vstack(out)


def draw_path(ax, sq, sol, color, title):
    n = sq.n
    # rejilla polar tenue
    tt = np.linspace(0, 2 * math.pi, 160)
    for rr in sq.r:
        ax.plot(rr * np.cos(tt), rr * np.sin(tt), color="0.88", lw=0.5, zorder=0)
    for ang in sq.a:
        ax.plot([0, sq.r[-1] * math.cos(ang)], [0, sq.r[-1] * math.sin(ang)],
                color="0.88", lw=0.5, zorder=0)

    theta = np.array([sq.a[v] for v in sol])
    X, Y = polar_2_cartesian(theta, sq.r)
    curve = catmull_rom(np.c_[X, Y])
    ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.7, zorder=1)
    ax.scatter(X, Y, s=26, c="black", zorder=3)

    lim = sq.r[-1] * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8)


def main():
    n = 7
    sq = SpiralQueens(n)
    sols = solve_queens(n)

    spirals, others = [], []
    for s in sols:
        d, c = min_degree_poly(s)
        (spirals if d == 1 else others).append((s, d, c))
    others.sort(key=lambda e: (e[1], e[0]))
    print(f"grado 1 (espirales): {len(spirals)}   grado>=2 (sin espiral): {len(others)}")

    color = {1: "#1f77b4", 4: "#ff7f0e", 5: "#d62728"}
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))

    # fila 0: 4 espirales de referencia (grado 1)
    for k in range(4):
        s, d, c = spirals[k * 7]  # espaciadas
        draw_path(axes[0, k], sq, s, color[1], f"ESPIRAL (grado 1)\n{fmt_poly(c)}")

    # filas 1-3: las 12 sin espiral
    for idx, (s, d, c) in enumerate(others):
        r, col = 1 + idx // 4, idx % 4
        draw_path(axes[r, col], sq, s, color[d], f"grado {d}\n{fmt_poly(c)}")

    fig.suptitle("El 'camino' de las 40 soluciones de las 7 reinas\n"
                 "fila 1: espirales (grado 1)   ·   filas 2-4: las 12 'sin espiral' (grado 4 naranja, 5 rojo)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("nonspiral_paths.png", dpi=130)
    print("Guardado: nonspiral_paths.png")


if __name__ == "__main__":
    main()
