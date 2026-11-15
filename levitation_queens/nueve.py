"""
Las nueve toroidales sin descripcion (una de p=17, ocho de p=19): tablero y forma circular.
Forma circular: anillo = fila (de dentro a fuera), rayo = columna. Sin espirales.
Uso: ../venv/bin/python nueve.py   -> fig_nueve.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASOS = [(17, [20]), (19, [13, 14, 15, 16, 17, 18, 21, 22])]


def tablero(ax, T, p, titulo):
    for r in range(p):
        for c in range(p):
            if (r + c) % 2: ax.add_patch(plt.Rectangle((c, r), 1, 1, fc="0.9", ec="none"))
    ax.scatter(np.array(T) + .5, np.arange(p) + .5, s=30, c="black", zorder=3)
    ax.set_xlim(0, p); ax.set_ylim(0, p); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titulo, fontsize=9)
    ax.set_xlabel(str(list(T)), fontsize=6.5)


def circular(ax, T, p):
    rad = lambda x: 2 + np.asarray(x, float)
    tt = np.linspace(0, 2 * np.pi, 200)
    for r in range(p):
        ax.plot(rad(r) * np.cos(tt), rad(r) * np.sin(tt), color="0.85", lw=0.5, zorder=0)
    for c in range(p):
        a = 2 * np.pi * c / p
        ax.plot([rad(0) * np.cos(a), rad(p - 1) * np.cos(a)], [rad(0) * np.sin(a), rad(p - 1) * np.sin(a)],
                color="0.85", lw=0.5, zorder=0)
    th = 2 * np.pi * np.array(T) / p
    ax.scatter(rad(range(p)) * np.cos(th), rad(range(p)) * np.sin(th), s=30, c="black", zorder=3)
    lim = rad(p - 1) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")


if __name__ == "__main__":
    sols = []
    for p, ids in CASOS:
        O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
        sols += [(p, i, O[i]) for i in ids]
    fig, axs = plt.subplots(3, 6, figsize=(18, 10.5))
    for n, (p, i, o) in enumerate(sols):
        f, c = divmod(n, 3)
        tablero(axs[f][2 * c], o["rep"], p, f"p={p}, solucion {i}: {o['clase']} helices")
        circular(axs[f][2 * c + 1], o["rep"], p)
        print(p, i, o["clase"], o["rep"])
    fig.suptitle("Las nueve toroidales sin descripcion: tablero (fila 0 abajo) y forma circular (fila 0 = anillo interior)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    fig.savefig("fig_nueve.png", dpi=110)
    print("-> fig_nueve.png")
