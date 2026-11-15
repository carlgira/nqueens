"""
Dibuja el mejor empaquetamiento esferico de las reinas frente al optimo de
Thomson (bipiramide pentagonal para N=7), y la curva de energia E(k).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from spiral_queens import SpiralQueens
from sphere_packing import sphere_points, thomson_energy, THOMSON_MIN, best_k


def unit_sphere(ax, color="0.9"):
    u = np.linspace(0, 2 * math.pi, 40)
    v = np.linspace(0, math.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.25, linewidth=0, zorder=0)


def pentagonal_bipyramid():
    pts = [(0, 0, 1), (0, 0, -1)]
    for j in range(5):
        ang = 2 * math.pi * j / 5
        pts.append((math.cos(ang), math.sin(ang), 0))
    return np.array(pts)


def spiral_curve_on_sphere(sq, sol, k, num=400):
    n = sq.n
    m = ((sol[1] - sol[0]) % n)
    b = sol[0] % n
    jj = np.linspace(0, n - 1, num)
    t = 2 * math.pi * (m * jj + b) / n
    f = k * jj * math.pi / n
    return np.c_[np.cos(t), np.sin(t) * np.cos(f), np.sin(t) * np.sin(f)]


def main(n=7):
    sq = SpiralQueens(n)
    labels = sq.init_sols_label()
    ks = np.linspace(0.05, 4.0, 400)

    # mejor solucion
    best = None
    for sol, d, i, dr in labels:
        s = [int(x) for x in sol]
        k, e = best_k(sq, s, ks)
        if best is None or e < best[0]:
            best = (e, k, s, d, i, dr)
    e_best, k_best, sol_best, d, i, dr = best
    pts = sphere_points(sq, sol_best, k_best)
    curve = spiral_curve_on_sphere(sq, sol_best, k_best)

    opt = THOMSON_MIN[n]
    bip = pentagonal_bipyramid()

    fig = plt.figure(figsize=(13, 4.5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    unit_sphere(ax1)
    ax1.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#1f77b4", lw=1.5)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="k", s=45, depthshade=False)
    ax1.set_title(f"Reinas en espiral esferica\nE={e_best:.3f} (k={k_best:.2f})")
    ax1.set_box_aspect((1, 1, 1)); ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    unit_sphere(ax2)
    ax2.scatter(bip[:, 0], bip[:, 1], bip[:, 2], c="#d62728", s=45, depthshade=False)
    ax2.set_title(f"Optimo de Thomson\n(bipiramide pentagonal) E={opt:.3f}")
    ax2.set_box_aspect((1, 1, 1)); ax2.set_axis_off()

    ax3 = fig.add_subplot(1, 3, 3)
    energies = [thomson_energy(sphere_points(sq, sol_best, kk)) for kk in ks]
    ax3.plot(ks, energies, color="#1f77b4")
    ax3.axhline(opt, color="#d62728", ls="--", label=f"optimo Thomson {opt:.2f}")
    ax3.plot(k_best, e_best, "ko")
    ax3.set_xlabel("k (inclinacion)"); ax3.set_ylabel("energia E")
    ax3.set_title("E(k) de la mejor solucion"); ax3.legend(); ax3.set_ylim(opt - 1, opt + 12)

    fig.suptitle(f"{n} reinas empaquetadas en la esfera vs optimo de Thomson")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"sphere_packing_n{n}.png"
    fig.savefig(out, dpi=130)
    print(f"Guardado: {out}  sol={sol_best} k={k_best:.3f} E={e_best:.4f} "
          f"(+{100*(e_best-opt)/opt:.2f}% vs optimo)")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
