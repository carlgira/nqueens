"""
Como lucen los niveles (clases) de toroidales dibujados como espirales.

Rejilla polar: fila r = anillo r, columna c = rayo c. La recta T(r) = a*r + c (mod p)
es la espiral logaritmica angulo = 2*pi*(a*x + c)/p sobre el anillo continuo x.
Nivel 0 = una espiral; nivel 1 = dos espirales que se cruzan en una reina y se reparten
las filas por clases de H; clase 3 (HAZ 2+1) = la misma espiral dos veces (girada) + otra.

Uso: ../venv/bin/python espirales_niveles.py [p]   -> fig_niveles_espirales_p{p}.png
"""
import sys
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["#1f77b4", "#d62728", "#2ca02c"]


def signo(a, p):
    return a if a <= p // 2 else a - p


def subgrupo(p, gs):
    H, fr = {1}, [1]
    while fr:
        x = fr.pop()
        for g in gs:
            y = x * g % p
            if y not in H:
                H.add(y); fr.append(y)
    return H


def toroidal(T, p):
    return all(len({(T[r] + s * r) % p for r in range(p)}) == p for s in (0, 1, -1))


def nivel1(p, k_obj, r0, c0):
    """Primera pareja (alfa, beta) con k clases; devuelve T y la recta de cada fila."""
    inv = lambda x: pow(x, p - 2, p)
    pares = sorted(combinations(range(2, p - 1), 2),
                   key=lambda ab: abs(signo(ab[0], p)) + abs(signo(ab[1], p)))
    for a, b in pares:
        H = subgrupo(p, [b * inv(a) % p, (a + 1) * inv(b + 1) % p, (a - 1) * inv(b - 1) % p])
        if (p - 1) // len(H) != k_obj:
            continue
        S = set(H)                                   # una clase toma beta, el resto alfa
        T, cual = [], []
        for r in range(p):
            x = (r - r0) % p
            j = 1 if x in S else 0
            T.append((c0 + (b if j else a) * x) % p)
            cual.append(-1 if x == 0 else j)
        assert toroidal(T, p)
        return T, cual, [(a, (c0 - a * r0) % p), (b, (c0 - b * r0) % p)]
    return None


def clase3(p):
    """Primera toroidal de clase 3 del volcado, con su cobertura (a, a, -a), a^2 = -1."""
    i = next(a for a in range(2, p) if a * a % p == p - 1)
    for linea in open(f"data/clases_toro_p{p}.txt"):
        v = list(map(int, linea.split()))
        if v[0] != 3:
            continue
        T = v[1:]
        rectas = {}
        for r in range(p):
            for a in (i, p - i):
                rectas.setdefault((a, (T[r] - a * r) % p), []).append(r)
        gr = sorted(rectas.items(), key=lambda kv: -len(kv[1]))[:3]
        if sum(len(f) for _, f in gr) < p:
            continue
        cual = [next(j for j, (_, f) in enumerate(gr) if r in f) for r in range(p)]
        return T, cual, [ac for ac, _ in gr]
    return None


def dibuja(ax, p, T, cual, rectas, titulo):
    q = 1.13
    rad = lambda x: q ** np.asarray(x, float)
    tt = np.linspace(0, 2 * np.pi, 200)
    for r in range(p):
        ax.plot(rad(r) * np.cos(tt), rad(r) * np.sin(tt), color="0.88", lw=0.5, zorder=0)
    for c in range(p):
        ang = 2 * np.pi * c / p
        ax.plot([0, rad(p - 1) * np.cos(ang)], [0, rad(p - 1) * np.sin(ang)], color="0.88", lw=0.5, zorder=0)
    x = np.linspace(0, p - 1, 1500)
    for j, (a, c) in enumerate(rectas):
        th = 2 * np.pi * (signo(a, p) * x + c) / p
        ax.plot(rad(x) * np.cos(th), rad(x) * np.sin(th), color=COL[j], lw=1.4, alpha=0.75, zorder=1)
    for r in range(p):
        th = 2 * np.pi * T[r] / p
        ax.scatter(rad(r) * np.cos(th), rad(r) * np.sin(th), s=46, zorder=3,
                   c="black" if cual[r] < 0 else COL[cual[r]], edgecolors="black", linewidths=0.6)
    ax.set_title(titulo, fontsize=9)
    lim = rad(p - 1) * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")


def main(p=13):
    r0, c0 = p // 2, 0
    paneles = []
    a = 2
    paneles.append(([(a * r) % p for r in range(p)], [0] * p, [(a, 0)],
                    f"Nivel 0: una espiral\npendiente {a}"))
    for k in sorted({k for k in range(2, p) if (p - 1) % k == 0}):
        res = nivel1(p, k, r0, c0)
        if res and len(paneles) < 3:
            T, cual, rectas = res
            n = [cual.count(0), cual.count(1)]
            paneles.append((T, cual, rectas,
                            f"Nivel 1 (k={k}): dos espirales\npendientes {signo(rectas[0][0], p)} y "
                            f"{signo(rectas[1][0], p)}, {n[0]}+{n[1]} reinas + cruce"))
    try:
        res = clase3(p)
    except FileNotFoundError:
        res = None
    if res:
        T, cual, rectas = res
        paneles.append((T, cual, rectas,
                        "Clase 3 (HAZ 2+1): tres espirales\npendientes "
                        + ", ".join(str(signo(a, p)) for a, _ in rectas)))
    fig, axs = plt.subplots(1, len(paneles), figsize=(4.6 * len(paneles), 5.6))
    for ax, (T, cual, rectas, tit) in zip(np.atleast_1d(axs), paneles):
        assert toroidal(T, p)
        dibuja(ax, p, T, cual, rectas, tit)
        print(tit.replace("\n", " | "), "T =", T)
    fig.suptitle(f"Toroidales de p={p} como espirales (color = espiral que lleva cada reina; negro = cruce)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = f"fig_niveles_espirales_p{p}.png"
    fig.savefig(out, dpi=130)
    print("->", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 13)
