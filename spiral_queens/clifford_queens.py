"""
Las N reinas en el TORO DE CLIFFORD (4D), proyectado a R³.

Cada celda (r,c) -> punto en el toro de Clifford de S³ ⊂ R⁴:
    (cos u, sin u, cos v, sin v)/√2,   u=2πr/N, v=2πc/N
Las 4 restricciones = 4 familias de círculos:
    filas (u const), columnas (v const), diagonal (u+v const = fibra de Hopf),
    anti-diagonal (u-v const = fibra de anti-Hopf).
Una solucion lineal σ(i)=m·i+b es un nudo torico (1,m): un geodesico cerrado que
toca cada familia una vez. La familia de lazos = los distintos m validos.
"""

import numpy as np
import matplotlib.pyplot as plt

S2 = np.sqrt(2.0)


def clifford_to_r3(u, v):
    """Toro de Clifford (S³) -> R³ por proyeccion estereografica desde (0,0,0,1)."""
    a = np.cos(u) / S2
    b = np.sin(u) / S2
    c = np.cos(v) / S2
    d = np.sin(v) / S2
    k = 1.0 - d
    return a / k, b / k, c / k


def torus_surface(ax, N):
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, 2 * np.pi, 80)
    U, V = np.meshgrid(u, v)
    X, Y, Z = clifford_to_r3(U, V)
    ax.plot_surface(X, Y, Z, color="0.85", alpha=0.15, linewidth=0, zorder=0)


def knot(ax, N, m, b, color, lw=2.0):
    s = np.linspace(0, N, 600)
    u = 2 * np.pi * s / N
    v = 2 * np.pi * (m * s + b) / N
    X, Y, Z = clifford_to_r3(u, v)
    ax.plot(X, Y, Z, color=color, lw=lw, zorder=2)


def grid_points(ax, N, sol=None, color="black"):
    for r in range(N):
        for c in range(N):
            x, y, z = clifford_to_r3(2 * np.pi * r / N, 2 * np.pi * c / N)
            ax.scatter(x, y, z, s=6, c="0.5", zorder=1)
    if sol is not None:
        for r, c in enumerate(sol):
            x, y, z = clifford_to_r3(2 * np.pi * r / N, 2 * np.pi * c / N)
            ax.scatter(x, y, z, s=45, c=color, zorder=3)


def setup(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=28, azim=-55)
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-2.2, 2.2)


def main(N=7):
    fig = plt.figure(figsize=(13, 6.2))

    # panel 1: una solucion (m=2) como nudo torico + una diagonal (fibra de Hopf)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    torus_surface(ax1, N)
    sol2 = [(2 * r) % N for r in range(N)]
    grid_points(ax1, N, sol2, color="#1f77b4")
    knot(ax1, N, 2, 0, "#1f77b4")
    # una diagonal r+c=const  => u+v=const  (fibra de Hopf, círculo de Villarceau)
    t = np.linspace(0, 2 * np.pi, 400)
    Xd, Yd, Zd = clifford_to_r3(t, (2 * np.pi * 3 / N) - t)
    ax1.plot(Xd, Yd, Zd, color="#d62728", lw=1.6, ls="--", zorder=2)
    setup(ax1, "solución m=2 (nudo tórico azul)\ndiagonal r+c=3 = fibra de Hopf (rojo punteado)")

    # panel 2: la FAMILIA de lazos (varios m) sobre el mismo toro
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    torus_surface(ax2, N)
    cmap = plt.get_cmap("turbo")
    slopes = [2, 3, 4, 5]
    for i, m in enumerate(slopes):
        knot(ax2, N, m, 0, cmap(i / (len(slopes) - 1)), lw=1.7)
    setup(ax2, f"familia de lazos: nudos tóricos (1,m), m={slopes}\ncada m válido = una solución N-reinas")

    fig.suptitle("Las 7 reinas en el toro de Clifford (4D → R³): "
                 "4 restricciones = 4 familias de círculos", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("clifford_queens.png", dpi=130)
    print("Guardado: clifford_queens.png")


if __name__ == "__main__":
    main(7)
