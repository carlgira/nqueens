"""
Numero de enlace (linking number) entre nudos-solucion en el toro de Clifford.

Cada solucion lineal σ(i)=m·i+b es un nudo torico (1,m). Dos de pendiente distinta
se cruzan en 1 reina (2 rectas del plano afin se cortan en 1 punto), asi que para
medir el enlace las ponemos en toros ANIDADOS (theta = pi/2 ± eps) y calculamos la
integral de Gauss. El resultado (entero) mide cuanto se enredan dos soluciones.
"""

import numpy as np
import matplotlib.pyplot as plt


def clifford_point(theta, u, v):
    a = np.cos(theta / 2) * np.cos(u)
    b = np.cos(theta / 2) * np.sin(u)
    c = np.sin(theta / 2) * np.cos(v)
    d = np.sin(theta / 2) * np.sin(v)
    k = 1.0 - d
    return np.stack([a / k, b / k, c / k], axis=-1)


def solution_curve(N, m, b, theta, n=500):
    s = np.linspace(0, N, n, endpoint=False)
    return clifford_point(theta, 2 * np.pi * s / N, 2 * np.pi * (m * s + b) / N)


def gauss_linking(C1, C2):
    dr1 = np.roll(C1, -1, axis=0) - C1
    dr2 = np.roll(C2, -1, axis=0) - C2
    mid1 = C1 + dr1 / 2
    mid2 = C2 + dr2 / 2
    R = mid1[:, None, :] - mid2[None, :, :]          # (M,M,3)
    cross = np.cross(dr1[:, None, :], dr2[None, :, :])
    num = np.sum(R * cross, axis=-1)
    denom = np.linalg.norm(R, axis=-1) ** 3 + 1e-12
    return np.sum(num / denom) / (4 * np.pi)


def main(N=7):
    slopes = [2, 3, 4, 5]
    eps = 0.35
    print(f"N={N}: número de enlace entre soluciones de pendiente m y m'\n")
    print("       " + "".join(f"m'={mp:2d}   " for mp in slopes))
    for m in slopes:
        C1 = solution_curve(N, m, 0, np.pi / 2 - eps)
        row = []
        for mp in slopes:
            C2 = solution_curve(N, mp, 1, np.pi / 2 + eps)  # b distinto -> curvas distintas
            row.append(gauss_linking(C1, C2))
        print(f"  m={m} | " + "".join(f"{v:6.2f} " for v in row))

    # figura: dos soluciones enlazadas (m=2 y m=3) en toros anidados
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    C1 = solution_curve(N, 2, 0, np.pi / 2 - eps, n=800)
    C2 = solution_curve(N, 3, 1, np.pi / 2 + eps, n=800)
    lk = gauss_linking(C1, C2)
    ax.plot(*C1.T, color="#1f77b4", lw=2, label="solución m=2")
    ax.plot(*C2.T, color="#d62728", lw=2, label="solución m=3")
    ax.set_title(f"Dos soluciones enlazadas en el toro de Clifford\n"
                 f"número de enlace ≈ {lk:.2f}")
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=26, azim=-55); ax.legend()
    fig.tight_layout(); fig.savefig("linking.png", dpi=130)
    print("\nGuardado: linking.png")


if __name__ == "__main__":
    main(7)
