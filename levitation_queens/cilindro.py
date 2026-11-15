"""
Toroidales por clase dibujadas sobre un cilindro (fila = altura, columna = angulo).

Cada recta T(r) = a*r + c es una helice; dos rectas paralelas son LA MISMA helice girada.
Un ejemplo por (clase, tipo) sacado de data/orbitas_p{p}.json, con una cobertura minima.
Arriba el cilindro, abajo el mismo cilindro desenrollado.

Uso: ../venv/bin/python cilindro.py [p]   -> fig_cilindro_p{p}.png
"""
import sys, json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orbitas import min_covers
from tipos import tipo

COL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]


def signo(a, p):
    return a if a <= p // 2 else a - p


def ejemplos(p, cmax):
    out = {}
    for o in sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: o["clase"]):
        if o["clase"] > cmax:
            continue
        T = o["rep"]
        k, covs = min_covers(T, p)
        for cv in covs:
            t = (tipo(cv, T, p) or "otra").split(" [")[0] if k >= 3 else ""
            out.setdefault((k, t), (T, cv))
    return sorted(out.items())


def colores(cv):
    """mismo color para rectas paralelas (misma helice girada)."""
    sl = sorted({a for a, c, m in cv})
    return [COL[sl.index(a) % len(COL)] for a, c, m in cv]


def panel(ax3, ax2, p, T, cv, titulo):
    cs = colores(cv)
    x = np.linspace(0, p - 1, 1200)
    for (a, c, m), col in zip(cv, cs):
        col_cont = signo(a, p) * x + c
        th = 2 * np.pi * col_cont / p
        lejos = np.sin(th) > 0
        for mask, alpha in ((lejos, 0.22), (~lejos, 0.9)):
            xs = np.where(mask, np.cos(th), np.nan)
            ax3.plot(xs, np.sin(th), x, color=col, lw=1.3, alpha=alpha)
        y = np.mod(col_cont + 0.5, p) - 0.5
        y[np.abs(np.diff(y, prepend=y[0])) > p / 2] = np.nan
        ax2.plot(y, x, color=col, lw=1.2, alpha=0.8)
    tt = np.linspace(0, 2 * np.pi, 100)
    for z in (0, p - 1):
        ax3.plot(np.cos(tt), np.sin(tt), z, color="0.7", lw=0.6)
    for r in range(p):
        j = next(i for i, (a, c, m) in enumerate(cv) if m >> r & 1)
        varias = sum(m >> r & 1 for a, c, m in cv) > 1
        col = "black" if varias else cs[j]
        th = 2 * np.pi * T[r] / p
        ax3.scatter(np.cos(th), np.sin(th), r, s=34 if np.sin(th) <= 0 else 14, c=col,
                    edgecolors="black", linewidths=0.5, alpha=1 if np.sin(th) <= 0 else 0.45, depthshade=False)
        ax2.scatter(T[r], r, s=34, c=col, edgecolors="black", linewidths=0.5, zorder=3)
    ax3.set_box_aspect((1, 1, 2.1)); ax3.view_init(elev=12, azim=-90); ax3.axis("off")
    ax3.set_title(titulo, fontsize=9)
    ax2.set_xlim(-0.6, p - 0.4); ax2.set_ylim(-0.6, p - 0.4); ax2.set_aspect("equal")
    ax2.set_xticks(range(p)); ax2.set_yticks(range(p)); ax2.tick_params(labelsize=5)
    ax2.grid(color="0.9", lw=0.5); ax2.set_axisbelow(True)


def main(p=17, cmax=5):
    ejs = ejemplos(p, cmax)
    n = len(ejs)
    fig = plt.figure(figsize=(3.5 * n, 10))
    for i, ((k, t), (T, cv)) in enumerate(ejs):
        sl = Counter(signo(a, p) for a, c, m in cv)
        pend = ", ".join(f"{a}" + (f" (x{v})" if v > 1 else "") for a, v in sorted(sl.items()))
        tit = f"clase {k}" + (f": {t}" if t else "") + f"\npendientes {pend}"
        panel(fig.add_subplot(2, n, i + 1, projection="3d"), fig.add_subplot(2, n, n + i + 1), p, T, cv, tit)
        print(k, t, pend, T)
    fig.suptitle(f"Toroidales de p={p} sobre un cilindro: una helice por recta; mismo color = misma helice girada; "
                 f"negro = reina en un cruce", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"fig_cilindro_p{p}.png"
    fig.savefig(out, dpi=110)
    print("->", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 17, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
