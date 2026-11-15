"""
Empaquetamiento en la esfera (problema de Thomson) a partir de las soluciones
espirales de las N reinas.

Al proyectar las reinas a la esfera unidad, cada una queda determinada por su
angulo theta = 2*pi*sol[j]/N y su inclinacion phi = k*j*pi/N (la norma se
conserva, asi que caen exactas sobre la esfera):

    p_j = (cos t, sin t*cos f, sin t*sin f)

Como sol[j] es lineal en j, los N puntos forman una espiral esferica.
Medimos la energia de Thomson  E = sum_{i<j} 1/|p_i - p_j|  y la comparamos con:
  - el minimo conocido de Thomson para N puntos,
  - una nube aleatoria de N puntos (baseline).
"""

import math
import numpy as np
from spiral_queens import SpiralQueens

# Minimos de energia de Thomson conocidos (sum 1/r), para referencia.
THOMSON_MIN = {5: 6.474691495, 7: 14.452977414, 11: 40.596450788, 13: 58.853230612}


def sphere_points(sq, sol, k):
    n = sq.n
    t = np.array([2 * math.pi * v / n for v in sol])      # theta_j
    f = np.array([k * j * math.pi / n for j in range(n)])  # phi_j
    x = np.cos(t)
    y = np.sin(t) * np.cos(f)
    z = np.sin(t) * np.sin(f)
    return np.c_[x, y, z]


def thomson_energy(pts):
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), 1)
    dd = d[iu]
    if np.any(dd < 1e-9):
        return np.inf
    return np.sum(1.0 / dd)


def random_baseline(n, trials=4000):
    rng = np.random.default_rng(0)
    best = np.inf
    acc = 0.0
    for _ in range(trials):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        e = thomson_energy(v)
        acc += e
        best = min(best, e)
    return acc / trials, best


def best_k(sq, sol, ks):
    energies = [thomson_energy(sphere_points(sq, sol, k)) for k in ks]
    i = int(np.argmin(energies))
    return ks[i], energies[i]


def main(n=7):
    sq = SpiralQueens(n)
    labels = sq.init_sols_label()
    ks = np.linspace(0.05, 4.0, 400)

    results = []
    for sol, d, i, dr in labels:
        s = [int(x) for x in sol]
        m = (s[1] - s[0]) % n
        k, e = best_k(sq, s, ks)
        results.append((e, k, m, d, i, dr, s))

    results.sort(key=lambda r: r[0])
    opt = THOMSON_MIN.get(n)
    avg_rand, best_rand = random_baseline(n)

    print(f"N={n}   optimo Thomson = {opt}")
    print(f"aleatorio: media={avg_rand:.4f}  mejor de 4000={best_rand:.4f}\n")
    print("  E(min)   k     m  (d,a)   solucion            gap vs optimo")
    for e, k, m, d, i, dr, s in results:
        gap = 100 * (e - opt) / opt if opt else float('nan')
        tag = "s1" if dr == 1 else "s2"
        print(f"  {e:7.4f}  {k:4.2f}  {m}  (d{d},a{i},{tag})  {str(s):20s} +{gap:5.2f}%")

    best = results[0]
    print(f"\nMejor: E={best[0]:.4f} con k={best[1]:.2f}  (m={best[2]}) "
          f"-> a {100*(best[0]-opt)/opt:.2f}% del optimo de Thomson.")
    return results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    main(n)
