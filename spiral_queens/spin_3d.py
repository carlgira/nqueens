"""
GIF rotatorio de una solucion espiral en 3D. Aqui se ve de verdad como cada
anillo esta en su propio plano y como la espiral se tuerce fuera del plano.

Uso:
    python spin_3d.py               # N=7, solucion #14, k=1
    python spin_3d.py 7 14 1        # N, indice, k
    python spin_3d.py 7 14 2        # k=2 -> mas inclinacion acumulada
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from spiral_queens import SpiralQueens, is_prime
from plot_solutions_3d import draw_solution_3d


def main(n=7, index=14, k=1.0, frames=72):
    sq = SpiralQueens(n)
    labels = sq.init_sols_label()
    sol, d, i, direction = labels[index % len(labels)]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_solution_3d(ax, sq, sol, d, i, direction, k=k)
    ax.set_axis_off()
    tag = "s1" if direction == 1 else "s2"
    fig.suptitle(f"{n}-reinas 3D  sol #{index}  (d={d}, a={i}, {tag}, k={k})")

    def update(f):
        ax.view_init(elev=20, azim=f * 360 / frames)
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)
    out = f"spin_3d_n{n}_one{index}_k{int(k)}.gif"
    anim.save(out, writer=PillowWriter(fps=18))
    print(f"Guardado: {out}  ->  sol = {list(sol)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    if not is_prime(n):
        raise SystemExit(f"N={n} no es primo.")
    main(n, index, k)
