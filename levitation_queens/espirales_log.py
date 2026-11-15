"""
Nivel 1 (dos espirales) redibujado en la rejilla "de potencias".

Rejilla normal: anillo = fila, rayo = columna. Rejilla de potencias: se quita la reina del cruce
(va al centro) y se reordenan anillos y rayos por potencias de una raiz primitiva g:
anillo t = fila a distancia g^t del cruce, rayo j = columna a distancia g^j del cruce.
Ahi las dos espirales tienen la MISMA velocidad (una vuelta) y solo estan giradas una respecto
a la otra, y las reinas se turnan con periodo k: es la alternancia que en la rejilla normal no se ve.

Uso: ../venv/bin/python espirales_log.py [p]   -> fig_espirales_log_p{p}.png
"""
import sys
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from espirales_niveles import signo, subgrupo, toroidal, dibuja, COL


def raiz_primitiva(p):
    return next(g for g in range(2, p) if len(subgrupo(p, [g])) == p - 1)


def parejas(p):
    """Una pareja (alfa, beta) por cada k; ademas una que no sea de sentidos opuestos si la hay."""
    inv = lambda x: pow(x, p - 2, p)
    pares = sorted(combinations(range(2, p - 1), 2),
                   key=lambda ab: abs(signo(ab[0], p)) + abs(signo(ab[1], p)))
    vistos, out = set(), []
    for a, b in pares:
        H = subgrupo(p, [b * inv(a) % p, (a + 1) * inv(b + 1) % p, (a - 1) * inv(b - 1) % p])
        k = (p - 1) // len(H)
        clave = (k, (a + b) % p == 0)
        if k > 1 and clave not in vistos:
            vistos.add(clave); out.append((a, b, k, H))
    return sorted(out, key=lambda t: (t[2], (t[0] + t[1]) % p != 0))


def panel_log(ax, p, g, a, b, k, titulo):
    n = p - 1
    log = {pow(g, t, p): t for t in range(n)}
    q = 1.13
    rad = lambda x: 1.6 * q ** np.asarray(x, float)
    tt = np.linspace(0, 2 * np.pi, 200)
    for t in range(n):
        ax.plot(rad(t) * np.cos(tt), rad(t) * np.sin(tt), color="0.88", lw=0.5, zorder=0)
    for j in range(n):
        ang = 2 * np.pi * j / n
        ax.plot([0, rad(n - 1) * np.cos(ang)], [0, rad(n - 1) * np.sin(ang)], color="0.88", lw=0.5, zorder=0)
        ax.text(rad(n - 1) * 1.09 * np.cos(ang), rad(n - 1) * 1.09 * np.sin(ang), str(signo(pow(g, j, p), p)),
                fontsize=6, color="0.45", ha="center", va="center")
    tau = np.linspace(0, n - 1, 800)
    for j, m in enumerate((a, b)):
        th = 2 * np.pi * (tau + log[m]) / n
        ax.plot(rad(tau) * np.cos(th), rad(tau) * np.sin(th), color=COL[j], lw=1.4, alpha=0.75, zorder=1)
    for t in range(n):
        j = 1 if t % k == 0 else 0                       # la clase H (t multiplo de k) toma beta
        th = 2 * np.pi * (t + log[(a, b)[j]]) / n
        ax.scatter(rad(t) * np.cos(th), rad(t) * np.sin(th), s=46, c=COL[j], edgecolors="black",
                   linewidths=0.6, zorder=3)
    ax.scatter(0, 0, s=46, c="black", zorder=3)
    ax.set_title(titulo, fontsize=9)
    lim = rad(n - 1) * 1.16
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")


def main(p=13):
    g = raiz_primitiva(p)
    r0, c0 = p // 2, 0
    ps = parejas(p)
    fig, axs = plt.subplots(2, len(ps), figsize=(4.6 * len(ps), 10.4), squeeze=False)
    for col, (a, b, k, H) in enumerate(ps):
        T, cual = [], []
        for r in range(p):
            x = (r - r0) % p
            j = 1 if x in H else 0
            T.append((c0 + (b if j else a) * x) % p)
            cual.append(-1 if x == 0 else j)
        assert toroidal(T, p)
        orden = "".join("XAR"[c + 1] for c in cual)
        dibuja(axs[0][col], p, T, cual, [(a, (c0 - a * r0) % p), (b, (c0 - b * r0) % p)],
               f"Rejilla normal, pendientes {signo(a, p)} y {signo(b, p)} (k={k})\nanillos 0..{p-1}: {orden}")
        log = {pow(g, t, p): t for t in range(p - 1)}
        giro = (log[b] - log[a]) % (p - 1)
        turno = "".join("R" if t % k == 0 else "A" for t in range(p - 1))
        panel_log(axs[1][col], p, g, a, b, k,
                  f"Rejilla de potencias de {g}: misma espiral girada {giro}/{p-1} de vuelta\nturnos: {turno}")
        print(a, b, "k", k, "giro", giro, "T", T, orden, turno)
    fig.suptitle(f"Nivel 1 en p={p}: arriba como lo veniamos dibujando, abajo reordenando anillos y rayos "
                 f"por potencias de {g} (numeros = columna respecto al cruce)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"fig_espirales_log_p{p}.png"
    fig.savefig(out, dpi=120)
    print("->", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 13)
