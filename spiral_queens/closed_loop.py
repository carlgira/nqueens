"""
Lazo CERRADO (epiciclos de Fourier) como camino canonico de cada solucion.

Idea: cada reina es un numero complejo  z_j = r[j]·e^{i·a·sol[j]}  en el plano.
La secuencia (z_0..z_{N-1}) se toma como UN periodo -> su DFT da N coeficientes
de Fourier = N epiciclos (circulos rotando) que dibujan un lazo cerrado unico
que pasa por las N reinas. Esto:
  - elimina la ambiguedad de "muchos caminos" (el lazo band-limited es unico),
  - funciona igual para soluciones bonitas y feas,
  - da una regresion canonica: los coeficientes c_k.

Ademas, para las 28 espirales (grado 1) ocurre algo mas fuerte: en el plano
complejo  z_j = A·w^j  es una secuencia GEOMETRICA, es decir el cociente entre
reinas consecutivas  w = z_{j+1}/z_j  es CONSTANTE. Ese unico w (2 numeros
reales: crecimiento radial |w| y paso angular arg w) es la regresion minima.
Para las 12 restantes el cociente NO es constante -> hacen falta mas epiciclos.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from spiral_queens import SpiralQueens
from discover_queens_formula import solve_queens, min_degree_poly, fmt_poly


def complex_points(sq, sol):
    theta = np.array([sq.a[v] for v in sol])
    return sq.r * np.exp(1j * theta)


def fourier_loop(z, num=500):
    """Lazo cerrado (interpolacion trigonometrica) que pasa por los puntos z."""
    N = len(z)
    C = np.fft.fft(z)
    ks = np.fft.fftfreq(N, d=1.0 / N)      # frecuencias enteras con signo
    s = np.linspace(0, 1, num)
    Z = np.zeros(num, dtype=complex)
    for k in range(N):
        Z += C[k] * np.exp(2j * np.pi * ks[k] * s)
    return Z / N, C, ks


def draw_epicycles(ax, C, ks, s0=0.12):
    """Dibuja la cadena de epiciclos (circulos) en la fase s0."""
    N = len(C)
    order = sorted(range(N), key=lambda k: -abs(C[k]))
    center = 0 + 0j
    th = np.linspace(0, 2 * np.pi, 60)
    for k in order:
        rad = abs(C[k]) / N
        nxt = center + C[k] / N * np.exp(2j * np.pi * ks[k] * s0)
        if rad > 1e-9:
            ax.plot(center.real + rad * np.cos(th), center.imag + rad * np.sin(th),
                    color="0.6", lw=0.5)
            ax.plot([center.real, nxt.real], [center.imag, nxt.imag], color="0.5", lw=0.5)
        center = nxt


def draw_one(ax, sq, sol, degree, c_poly, epicycles=True):
    z = complex_points(sq, sol)
    Z, C, ks = fourier_loop(z)
    col = "#1f77b4" if degree == 1 else ("#ff7f0e" if degree == 4 else "#d62728")
    if epicycles:
        draw_epicycles(ax, C, ks)
    ax.plot(Z.real, Z.imag, color=col, lw=1.8, zorder=2)
    ax.scatter(z.real, z.imag, s=28, c="black", zorder=3)
    lim = sq.r[-1] * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"grado {degree}\n{fmt_poly(c_poly)}", fontsize=8)


def ratio_constancy(sq, sol):
    """Desviacion del cociente complejo z_{j+1}/z_j (0 si es geometrica = espiral)."""
    z = complex_points(sq, sol)
    w = z[1:] / z[:-1]
    return np.std(np.abs(w)), np.std(np.angle(w))


def main():
    n = 7
    sq = SpiralQueens(n)
    sols = solve_queens(n)

    tagged = []
    for s in sols:
        d, c = min_degree_poly(s)
        tagged.append((s, d, c))
    spirals = [t for t in tagged if t[1] == 1]
    others = sorted([t for t in tagged if t[1] > 1], key=lambda e: e[1])

    # --- reporte numerico: constancia del cociente vs grado ---
    print("Regresion minima z_j = A·w^j  (w = cociente entre reinas consecutivas)")
    print("grado | #sols | std|w| medio | std(arg w) medio")
    for d in sorted(set(t[1] for t in tagged)):
        grp = [t for t in tagged if t[1] == d]
        sm = np.mean([ratio_constancy(sq, s)[0] for s, _, _ in grp])
        sa = np.mean([ratio_constancy(sq, s)[1] for s, _, _ in grp])
        print(f"  {d}   |  {len(grp):2d}   |   {sm:.2e}   |   {sa:.2e}")

    # comprobar que para las espirales w recupera la pendiente m
    print("\nPara una espiral, arg(w)/(2π/N) recupera la pendiente m:")
    for s, d, c in spirals[:3]:
        z = complex_points(sq, s)
        w = np.mean(z[1:] / z[:-1])
        m_rec = (np.angle(w) / (2 * math.pi / n)) % n
        print(f"  sol={s}  |w|={abs(w):.4f} (crecimiento)  m≈{m_rec:.3f}")

    # --- figura: lazos cerrados por epiciclos ---
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    picks = [spirals[0], spirals[7], spirals[14], spirals[21],
             others[0], others[3], others[8], others[11]]
    for ax, (s, d, c) in zip(axes.reshape(-1), picks):
        draw_one(ax, sq, s, d, c)
    fig.suptitle("Lazo cerrado canónico por epiciclos de Fourier\n"
                 "fila 1: espirales (grado 1)  ·  fila 2: 'feas' (grado 4/5) — todas cierran igual",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("closed_loop.png", dpi=130)
    print("\nGuardado: closed_loop.png")


if __name__ == "__main__":
    main()
