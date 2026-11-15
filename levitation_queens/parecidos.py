"""
¿Se parecen entre si las toroidales sin descripcion de p=19?
 1. Mapa de calor: cuantas de las 8 tienen reina en cada casilla (tal como se dibujan), comparado
    con las otras 15 soluciones distintas dibujadas con el mismo criterio.
 2. Parecido real: maximo de reinas en comun entre dos soluciones, probando todas las copias
    (traslacion, escala, giro) de la segunda.
Uso: ../venv/bin/python parecidos.py   -> fig_parecidos.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orbitas import mats, apply

p = 19
SIN = [13, 14, 15, 16, 17, 18, 21, 22]
O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
reps = [np.array(o["rep"]) for o in O]


def calor(ids):
    H = np.zeros((p, p), int)
    for i in ids: H[np.arange(p), reps[i]] += 1
    return H


def copias(T):
    """todas las copias de T: (8(p-1) simetrias) x (p^2 traslaciones), como matriz."""
    out = []
    for M in mats(p):
        U = apply(M, T, p)
        for s in range(p):
            V = np.roll(U, -s)
            out.append((V[None, :] + np.arange(p)[:, None]) % p)
    return np.concatenate(out)


n = len(O)
comun = np.zeros((n, n), int)
for j in range(n):
    C = copias(reps[j])
    for i in range(n):
        a = (C == reps[i]).sum(axis=1)
        if i == j: a = a[a < p]
        comun[i, j] = a.max()
otros = [i for i in range(n) if i not in SIN]
dentro = [comun[i, j] for i in SIN for j in SIN if i != j]
fuera = [comun[i, j] for i in SIN for j in otros]
resto = [comun[i, j] for i in otros for j in otros if i != j]
print("reinas en comun (mejor copia), media / max:")
print("  entre las 8 sin descripcion:", np.mean(dentro).round(2), max(dentro))
print("  de las 8 con las otras 15:  ", np.mean(fuera).round(2), max(fuera))
print("  entre las otras 15:         ", np.mean(resto).round(2), max(resto))
print("coincidencias en el dibujo tal cual (misma casilla), media por pareja:")
tal = lambda A, B: np.mean([(reps[i] == reps[j]).sum() for i in A for j in B if i != j]).round(2)
print("  entre las 8:", tal(SIN, SIN), " 8 con las otras:", tal(SIN, otros), " entre las otras:", tal(otros, otros))
print("primeras filas de cada solucion:", [list(map(int, r[:4])) for r in reps])

fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
for ax, ids, tit in ((axs[0], SIN, "las 8 sin descripcion"), (axs[1], otros, "las otras 15")):
    H = calor(ids)
    im = ax.imshow(H, origin="lower", cmap="Blues", vmin=0, vmax=max(3, H.max()))
    for r in range(p):
        for c in range(p):
            if H[r, c] > 1: ax.text(c, r, H[r, c], ha="center", va="center", fontsize=7,
                                    color="white" if H[r, c] > H.max() / 2 else "black")
    ax.set_title(f"Casillas ocupadas: {tit}\n(numero = cuantas soluciones tienen reina ahi)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
im = axs[2].imshow(comun, cmap="viridis")
axs[2].set_title("Reinas en comun entre dos soluciones\n(eligiendo la mejor copia de la segunda)", fontsize=9)
axs[2].set_xticks(range(n)); axs[2].set_yticks(range(n)); axs[2].tick_params(labelsize=6)
for i in SIN:
    axs[2].get_xticklabels()[i].set_color("red"); axs[2].get_yticklabels()[i].set_color("red")
plt.colorbar(im, ax=axs[2], fraction=0.046)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("fig_parecidos.png", dpi=110); print("-> fig_parecidos.png")
