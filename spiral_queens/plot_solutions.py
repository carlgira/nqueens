"""
Dibuja TODAS las soluciones espirales de las N reinas para un N primo.

Cada solución se muestra sobre su rejilla polar: los anillos y rayos de la
rejilla, la espiral logarítmica que genera la solución, y las reinas (puntos
negros) que caen exactamente sobre la espiral.

Uso:
    python plot_solutions.py           # N = 7  -> grid de 28 soluciones
    python plot_solutions.py 11        # otro primo
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from spiral_queens import (
    SpiralQueens, phi, is_prime,
    spiral1, spiral1_inv, spiral2_inv, polar_2_cartesian,
)


def spiral_curve(sq, c, a, direction, num=600):
    """Puntos (X, Y) de la espiral entre el radio menor y el mayor de la rejilla."""
    r_dense = np.linspace(sq.r[0], sq.r[-1], num)
    inv = spiral1_inv if direction == 1 else spiral2_inv
    theta = np.array([inv(c, rr, a) for rr in r_dense])
    return polar_2_cartesian(theta, r_dense)


def draw_solution(ax, sq, sol, d, i, direction):
    c = math.pow(phi, 1 / (d * math.pi))
    a = sq.a[i]
    rmax = sq.r[-1]

    # Rejilla polar: anillos (radios) y rayos (ángulos), en gris tenue.
    tt = np.linspace(0, 2 * math.pi, 200)
    for rr in sq.r:
        ax.plot(rr * np.cos(tt), rr * np.sin(tt), color="0.85", lw=0.6, zorder=0)
    for ang in sq.a:
        ax.plot([0, rmax * math.cos(ang)], [0, rmax * math.sin(ang)],
                color="0.85", lw=0.6, zorder=0)

    # La espiral que genera la solución.
    color = "#1f77b4" if direction == 1 else "#d62728"
    X, Y = spiral_curve(sq, c, a, direction)
    ax.plot(X, Y, color=color, lw=1.6, zorder=1)

    # Las reinas: sobre el radio r[j] y el ángulo a[sol[j]].
    theta_q = np.array([sq.a[v] for v in sol])
    Xq, Yq = polar_2_cartesian(theta_q, sq.r)
    ax.scatter(Xq, Yq, s=28, c="black", zorder=3)

    tag = "s1" if direction == 1 else "s2"
    ax.set_title(f"d={d}  a={i}  {tag}", fontsize=8)
    lim = rmax * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def main(n=7):
    if not is_prime(n):
        raise SystemExit(f"N={n} no es primo; el método está pensado para primos.")

    sq = SpiralQueens(n)
    labels = sq.init_sols_label()
    total = len(labels)
    print(f"N={n}: {total} soluciones espirales "
          f"({sum(sq.validate(s) for s, *_ in labels)} válidas)")

    # Disposición del grid: lo más cuadrado posible.
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    # Para 28 preferimos 4 x 7.
    if total == 28:
        rows, cols = 4, 7

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axes = np.array(axes).reshape(-1)

    for ax, (sol, d, i, direction) in zip(axes, labels):
        draw_solution(ax, sq, sol, d, i, direction)
    for ax in axes[total:]:
        ax.axis("off")

    fig.suptitle(f"{total} soluciones espirales de las {n}-reinas "
                 f"(azul = espiral 1, rojo = espiral 2)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = f"spiral_queens_n{n}.png"
    fig.savefig(out, dpi=130)
    print(f"Guardado: {out}")
    plt.show()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    main(n)
