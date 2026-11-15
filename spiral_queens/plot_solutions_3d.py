"""
Versión 3D de las soluciones espirales de las N reinas.

Cada uno de los N anillos de la rejilla polar se inclina a su propio plano:
el anillo j se rota un ángulo  tilt_j = k * j * pi/N  alrededor del eje X.
Así los anillos se abren como un abanico (j=0 plano, ~vertical hacia j=N/2,
casi volteado en el exterior).

La espiral 2D que generaba cada solución se convierte en una curva 3D: para
un radio continuo r el ángulo sigue la misma espiral y la inclinación tilt(r)
interpola entre anillos. La curva pasa exactamente por las reinas.

Transformación de un punto (radio r, angulo theta) del anillo con inclinacion tilt:
    x = r*cos(theta)
    y = r*sin(theta)*cos(tilt)
    z = r*sin(theta)*sin(tilt)

Uso:
    python plot_solutions_3d.py            # N=7, grid 4x7 en 3D
    python plot_solutions_3d.py 7 grid     # idem
    python plot_solutions_3d.py 7 one 8    # una sola solucion (indice 8) en grande
    python plot_solutions_3d.py 7 one 8 2  # ... con k=2 (mas vueltas de inclinacion)
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from spiral_queens import (
    SpiralQueens, phi, is_prime,
    spiral1_inv, spiral2_inv,
)


def transform_3d(r, theta, tilt):
    """(radio, angulo, inclinacion) -> (x, y, z). Acepta escalares o arrays."""
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(tilt)
    z = r * np.sin(theta) * np.sin(tilt)
    return x, y, z


def ring_tilts(sq, k=1.0):
    """Inclinacion de cada anillo j: tilt_j = k * j * pi/N."""
    return np.array([k * j * math.pi / sq.n for j in range(sq.n)])


def draw_solution_3d(ax, sq, sol, d, i, direction, k=1.0, rings=True):
    n = sq.n
    tilts = ring_tilts(sq, k)
    color = "#1f77b4" if direction == 1 else "#d62728"

    # Anillos, cada uno inclinado a su propio plano.
    if rings:
        tt = np.linspace(0, 2 * math.pi, 160)
        for j in range(n):
            X, Y, Z = transform_3d(sq.r[j], tt, tilts[j])
            ax.plot(X, Y, Z, color="0.8", lw=0.5, zorder=0)

    # Curva 3D que generaliza la espiral (pasa por las reinas).
    c = math.pow(phi, 1 / (d * math.pi))
    a = sq.a[i]
    inv = spiral1_inv if direction == 1 else spiral2_inv
    r_dense = np.linspace(sq.r[0], sq.r[-1], 700)
    theta = np.array([inv(c, rr, a) for rr in r_dense])
    # inclinacion continua: interpola el indice de anillo segun el radio.
    frac = np.interp(r_dense, sq.r, np.arange(n))
    tilt_dense = k * frac * math.pi / n
    Xc, Yc, Zc = transform_3d(r_dense, theta, tilt_dense)
    ax.plot(Xc, Yc, Zc, color=color, lw=1.8)

    # Reinas.
    theta_q = np.array([sq.a[v] for v in sol])
    tilt_q = tilts  # reina j vive en el anillo j
    Xq, Yq, Zq = transform_3d(sq.r, theta_q, tilt_q)
    ax.scatter(Xq, Yq, Zq, s=25, c="black", depthshade=False)

    lim = sq.r[-1] * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    tag = "s1" if direction == 1 else "s2"
    ax.set_title(f"d={d} a={i} {tag}", fontsize=8)


def plot_grid(n=7, k=1.0):
    sq = SpiralQueens(n)
    labels = sq.init_sols_label()
    total = len(labels)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    if total == 28:
        rows, cols = 4, 7

    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for idx, (sol, d, i, direction) in enumerate(labels):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        draw_solution_3d(ax, sq, sol, d, i, direction, k=k)
        ax.set_axis_off()
        ax.view_init(elev=22, azim=-60)

    fig.suptitle(f"{total} soluciones espirales de las {n}-reinas en 3D "
                 f"(anillos inclinados k·j·pi/N, k={k})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"spiral_queens_3d_n{n}.png"
    fig.savefig(out, dpi=130)
    print(f"Guardado: {out}")
    plt.show()


def plot_one(n=7, index=0, k=1.0):
    sq = SpiralQueens(n)
    labels = sq.init_sols_label()
    sol, d, i, direction = labels[index % len(labels)]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_solution_3d(ax, sq, sol, d, i, direction, k=k)
    ax.view_init(elev=22, azim=-60)
    tag = "s1" if direction == 1 else "s2"
    fig.suptitle(f"{n}-reinas 3D  |  sol #{index}  (d={d}, a={i}, {tag}, k={k})")
    out = f"spiral_queens_3d_n{n}_one{index}.png"
    fig.savefig(out, dpi=140)
    print(f"Guardado: {out}  ->  sol = {list(sol)}")
    plt.show()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    mode = sys.argv[2] if len(sys.argv) > 2 else "grid"
    if not is_prime(n):
        raise SystemExit(f"N={n} no es primo.")
    if mode == "one":
        index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        k = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        plot_one(n, index, k)
    else:
        k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        plot_grid(n, k)
