"""
Fibración de Hopf: S³ ⊂ R⁴ se llena de círculos cerrados y enlazados, cada uno
sobre un punto de la base S². Aquí proyectamos estereográficamente S³ -> R³ para
verlos. Las fibras sobre un paralelo de S² forman un toro anidado (círculos de
Villarceau); todas están enlazadas entre sí (enlace de Hopf, Lk=1).
"""

import numpy as np
import matplotlib.pyplot as plt


def hopf_fiber(theta, phi, n=300):
    """Círculo de Hopf (en R⁴) sobre el punto de base (theta, phi), parametrizado por psi."""
    psi = np.linspace(0, 2 * np.pi, n)
    a = np.cos(theta / 2) * np.cos(psi)
    b = np.cos(theta / 2) * np.sin(psi)
    c = np.sin(theta / 2) * np.cos(phi + psi)
    d = np.sin(theta / 2) * np.sin(phi + psi)
    return a, b, c, d


def stereo(a, b, c, d):
    """Proyección estereográfica S³ -> R³ desde el polo (0,0,0,1)."""
    denom = 1.0 - d
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return a / denom, b / denom, c / denom


def main():
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # varios paralelos de la base S² (cada theta = un toro anidado)
    thetas = [0.62, 1.05, 1.5, 1.95, 2.4]
    cmap = plt.get_cmap("turbo")
    n_fibers = 22

    for ti, theta in enumerate(thetas):
        color = cmap(ti / (len(thetas) - 1))
        for phi in np.linspace(0, 2 * np.pi, n_fibers, endpoint=False):
            X, Y, Z = stereo(*hopf_fiber(theta, phi))
            # recortar tramos que se van al infinito cerca del polo
            R = np.sqrt(X**2 + Y**2 + Z**2)
            X[R > 8] = np.nan
            ax.plot(X, Y, Z, color=color, lw=0.8, alpha=0.8)

    ax.set_title("Fibración de Hopf (S³→S²) proyectada a R³\n"
                 "cada color = un paralelo de la base · cada curva = una fibra CERRADA, todas enlazadas")
    lim = 4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    fig.savefig("hopf_fibration.png", dpi=130)
    print("Guardado: hopf_fibration.png")


if __name__ == "__main__":
    main()
