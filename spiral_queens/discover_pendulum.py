"""
Descubrir la formula de un pendulo esferico a partir de sus trayectorias (SINDy).

Idea (Brunton-Proctor-Kutz, 2016):
  1) Se observan trayectorias: el estado (theta, phi, theta', phi') en el tiempo.
  2) Se estiman las aceleraciones (theta'', phi'') numericamente.
  3) Se arma una biblioteca de terminos candidatos Theta(estado).
  4) Regresion DISPERSA (least-squares con umbral) -> solo sobreviven los
     terminos reales: eso ES la ecuacion de movimiento.

Verdad de terreno (l=g=m=1), pendulo esferico en coords esfericas
(theta = angulo desde la vertical):
    theta'' =  sin(theta) cos(theta) phi'^2  -  sin(theta)
    phi''   = -2 (cos theta / sin theta) theta' phi'
(equivalente: se conservan la energia y L_z = sin^2(theta) phi').
"""

import math
import numpy as np

G_OVER_L = 1.0


def rhs(s):
    th, ph, dth, dph = s
    ddth = math.sin(th) * math.cos(th) * dph * dph - G_OVER_L * math.sin(th)
    ddph = -2.0 * math.cos(th) / math.sin(th) * dth * dph
    return np.array([dth, dph, ddth, ddph])


def simulate(s0, dt=0.004, n=30000):
    s = np.array(s0, float)
    traj = np.empty((n, 4))
    for i in range(n):
        traj[i] = s
        k1 = rhs(s); k2 = rhs(s + dt / 2 * k1)
        k3 = rhs(s + dt / 2 * k2); k4 = rhs(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return traj


def stlsq(Theta, y, thr, iters=20):
    """Sequential thresholded least squares."""
    xi = np.linalg.lstsq(Theta, y, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < thr
        xi[small] = 0.0
        big = ~small
        if not big.any():
            break
        xi_new = np.zeros_like(xi)
        xi_new[big] = np.linalg.lstsq(Theta[:, big], y, rcond=None)[0]
        if np.allclose(xi_new, xi):
            xi = xi_new; break
        xi = xi_new
    return xi


def report(name, xi, feats, thr):
    terms = [f"{c:+.3f}·{n}" for c, n in zip(xi, feats) if abs(c) > 1e-6]
    print(f"  {name} = " + "  ".join(terms) if terms else f"  {name} = 0")


def main():
    # Varias condiciones iniciales -> exploran mas del espacio de fases
    # (con una sola orbita el problema puede ser degenerado / no identificable).
    seeds = [(1.2, 0.0, 0.0, 1.5),
             (0.9, 0.0, 0.3, 1.0),
             (1.6, 0.0, -0.2, 0.8),
             (2.0, 0.0, 0.1, 0.6)]
    dt = 0.004
    trajs = [simulate(s0, dt=dt, n=15000) for s0 in seeds]
    traj = np.vstack(trajs)

    th, ph, dth, dph = traj.T
    # "Medimos" solo el estado; las aceleraciones se estiman por diferencias.
    ddth = np.gradient(dth, dt)
    ddph = np.gradient(dph, dt)
    # recortar bordes de cada segmento (gradient es peor en los extremos)
    keep = np.ones(len(th), bool)
    for b in range(0, len(th), 15000):
        keep[b:b + 3] = False; keep[b + 15000 - 3:b + 15000] = False
    th, dth, dph = th[keep], dth[keep], dph[keep]
    ddth, ddph = ddth[keep], ddph[keep]

    s, c = np.sin(th), np.cos(th)

    # --- biblioteca para theta'' ---
    feats_th = ["1", "sinθ", "cosθ", "sinθcosθ", "θ'^2", "φ'^2",
                "sinθ·φ'^2", "cosθ·φ'^2", "sinθcosθ·φ'^2", "sinθ·θ'^2"]
    Th_th = np.column_stack([
        np.ones_like(th), s, c, s * c, dth**2, dph**2,
        s * dph**2, c * dph**2, s * c * dph**2, s * dth**2,
    ])
    xi_th = stlsq(Th_th, ddth, thr=0.05)

    # --- biblioteca para phi'' ---
    feats_ph = ["1", "θ'φ'", "(cosθ/sinθ)·θ'φ'", "tanθ·θ'φ'", "sinθ·θ'φ'", "θ'^2", "φ'^2"]
    Th_ph = np.column_stack([
        np.ones_like(th), dth * dph, (c / s) * dth * dph, (s / c) * dth * dph,
        s * dth * dph, dth**2, dph**2,
    ])
    xi_ph = stlsq(Th_ph, ddph, thr=0.05)

    print("Ecuaciones recuperadas desde las trayectorias (SINDy):\n")
    report("θ''", xi_th, feats_th, 0.05)
    report("φ''", xi_ph, feats_ph, 0.05)
    print("\nVerdad de terreno:")
    print("  θ'' = -1.000·sinθ  +1.000·sinθcosθ·φ'^2")
    print("  φ'' = -2.000·(cosθ/sinθ)·θ'φ'")


if __name__ == "__main__":
    main()
