"""
Lazos caoticos / fractales sobre la esfera unidad.

Tres formas de conseguir un "lazo caotico" en S^2 (recordando que un flujo
continuo autonomo NO puede ser caotico en la esfera, por Poincare-Bendixson):

  A) KICKED TOP (trompo pateado): un MAPA discreto sobre la esfera. Espacio de
     fases mixto -> unas orbitas son lazos regulares (islas), otras llenan un
     mar caotico. Ejemplo canonico de caos clasico/cuantico (Haake).

  B) Atractor caotico 3D (Thomas) PROYECTADO radialmente a la esfera unidad:
     un lazo caotico continuo que se enrolla sobre S^2.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- A) kicked top
def kicked_top_orbit(X, Y, Z, p=math.pi / 2, k=2.6, steps=4000):
    """Itera el mapa clasico del trompo pateado sobre la esfera unidad.
    Precesion angulo p alrededor de X, luego torsion no lineal k*Z alrededor de Z.
    """
    xs, ys, zs = [], [], []
    for _ in range(steps):
        # precesion alrededor del eje X
        Y2 = Y * math.cos(p) - Z * math.sin(p)
        Z2 = Y * math.sin(p) + Z * math.cos(p)
        X2 = X
        # torsion (kick) alrededor del eje Z, angulo proporcional a Z2
        phi = k * Z2
        X3 = X2 * math.cos(phi) - Y2 * math.sin(phi)
        Y3 = X2 * math.sin(phi) + Y2 * math.cos(phi)
        Z3 = Z2
        X, Y, Z = X3, Y3, Z3
        xs.append(X); ys.append(Y); zs.append(Z)
    return np.array(xs), np.array(ys), np.array(zs)


# --------------------------------------------------- B) Thomas attractor -> S^2
def thomas_on_sphere(b=0.208, dt=0.02, steps=20000, x0=0.1, y0=0.0, z0=0.0):
    def f(s):
        x, y, z = s
        return np.array([math.sin(y) - b * x,
                         math.sin(z) - b * y,
                         math.sin(x) - b * z])
    s = np.array([x0, y0, z0], float)
    pts = np.empty((steps, 3))
    for i in range(steps):
        k1 = f(s); k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts[i] = s
    pts -= pts.mean(axis=0)                       # centrar
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)  # proyectar a la esfera
    return pts


def unit_sphere(ax, color="0.92"):
    u = np.linspace(0, 2 * math.pi, 40); v = np.linspace(0, math.pi, 20)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.2, linewidth=0, zorder=0)


def main():
    fig = plt.figure(figsize=(13, 6.5))

    # --- A) kicked top: varias condiciones iniciales sobre la esfera ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    unit_sphere(ax1)
    rng = np.random.default_rng(1)
    cmap = plt.get_cmap("turbo")
    n_orbits = 24
    for c in range(n_orbits):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        xs, ys, zs = kicked_top_orbit(*v, p=math.pi / 2, k=2.6, steps=1500)
        ax1.scatter(xs, ys, zs, s=1.2, color=cmap(c / n_orbits))
    ax1.set_title("A) Kicked top (mapa en S²)\nislas regulares + mar caótico")
    ax1.set_box_aspect((1, 1, 1)); ax1.set_axis_off(); ax1.view_init(22, -60)

    # --- B) Thomas proyectado: un lazo caotico continuo ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    unit_sphere(ax2)
    pts = thomas_on_sphere(steps=16000)
    ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=0.35, color="#1f77b4")
    ax2.set_title("B) Atractor de Thomas proyectado a S²\nlazo caótico continuo")
    ax2.set_box_aspect((1, 1, 1)); ax2.set_axis_off(); ax2.view_init(22, -60)

    fig.suptitle("Lazos caóticos sobre la esfera unidad", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("chaos_on_sphere.png", dpi=130)
    print("Guardado: chaos_on_sphere.png")


if __name__ == "__main__":
    main()
