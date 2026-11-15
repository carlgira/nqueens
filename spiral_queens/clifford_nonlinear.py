"""
Las 40 soluciones de las 7 reinas como lazos CERRADOS en el toro de Clifford (proy. R³).

Conectamos las reinas en orden de fila (r=0..N-1->0) por segmentos geodesicos del
toro plano. Las lineales (grado 1) salen nudos toricos (1,m) limpios; las de grado
alto salen lazos que se bambolean fuera de la geodesica. Mismo método para todas.
"""

import numpy as np
import matplotlib.pyplot as plt

from discover_queens_formula import solve_queens, min_degree_poly, fmt_poly

S2 = np.sqrt(2.0)


def clifford_r3(u, v):
    a = np.cos(u) / S2; b = np.sin(u) / S2
    c = np.cos(v) / S2; d = np.sin(v) / S2
    k = 1.0 - d
    return a / k, b / k, c / k


def torus_loop(sol, N, nper=50):
    U, V = [], []
    for r in range(N):
        u0, u1 = 2 * np.pi * r / N, 2 * np.pi * (r + 1) / N
        v0 = 2 * np.pi * sol[r] / N
        v1 = 2 * np.pi * sol[(r + 1) % N] / N
        dv = (v1 - v0 + np.pi) % (2 * np.pi) - np.pi        # wrap mas corto
        t = np.linspace(0, 1, nper, endpoint=False)
        U.append(u0 + (u1 - u0) * t)
        V.append(v0 + dv * t)
    return np.concatenate(U), np.concatenate(V)


def torus_surface(ax):
    u = np.linspace(0, 2 * np.pi, 60); v = np.linspace(0, 2 * np.pi, 60)
    Ug, Vg = np.meshgrid(u, v)
    X, Y, Z = clifford_r3(Ug, Vg)
    ax.plot_surface(X, Y, Z, color="0.85", alpha=0.12, linewidth=0)


def draw(ax, sol, N, color, title):
    torus_surface(ax)
    u, v = torus_loop(sol, N)
    X, Y, Z = clifford_r3(u, v)
    ax.plot(X, Y, Z, color=color, lw=1.6)
    for r in range(N):
        x, y, z = clifford_r3(2 * np.pi * r / N, 2 * np.pi * sol[r] / N)
        ax.scatter(x, y, z, s=14, c="black")
    ax.set_title(title, fontsize=7)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=30, azim=-55)
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-1.8, 1.8)


def main(N=7):
    sols = solve_queens(N)
    lin, non = [], []
    for s in sols:
        d, c = min_degree_poly(s)
        (lin if d == 1 else non).append((s, d, c))
    non.sort(key=lambda e: e[1])
    color = {1: "#1f77b4", 4: "#ff7f0e", 5: "#d62728"}

    fig = plt.figure(figsize=(12, 12))
    # fila 0: 4 lineales (nudos toricos limpios)
    for k in range(4):
        s, d, c = lin[k * 7]
        ax = fig.add_subplot(4, 4, k + 1, projection="3d")
        draw(ax, s, N, color[1], f"LINEAL (1,{(s[1]-s[0])%N})\n{fmt_poly(c)}")
    # filas 1-3: las 12 no lineales
    for idx, (s, d, c) in enumerate(non):
        ax = fig.add_subplot(4, 4, 5 + idx, projection="3d")
        draw(ax, s, N, color[d], f"grado {d}")

    fig.suptitle("Las 40 soluciones de las 7 reinas como lazos en el toro de Clifford\n"
                 "fila 1: lineales = nudos tóricos limpios · filas 2-4: las 12 'feas' se bambolean",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("clifford_nonlinear.png", dpi=125)
    print("Guardado: clifford_nonlinear.png")


if __name__ == "__main__":
    main(7)
