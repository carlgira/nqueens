"""
Mapa de complejidad de las 73712 soluciones de las 13 reinas segun DOS bases:
  - eje Y: Prony p  (nº de espirales ABSTRACTAS, w libre)
  - eje X: DFT-sparsity (nº de espirales REALES de rejilla, |w|=rho)
Datos medidos con el cruce prony_sweep + DFT. Muestra la anti-correlacion:
'simple' depende de la base.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ps = [1, 3, 4, 5, 6, 7]
dfts = [1, 8, 10, 11, 12]
M = np.array([
    [130,    0,    0,    0,     0],   # p=1
    [  0,    0,    0,    0,    18],   # p=3
    [  0,    0,    0,    0,     4],   # p=4
    [  0,    0,    0,    0,    30],   # p=5
    [  0,    0,    4,   24,    42],   # p=6
    [  0, 1352, 3162, 6812, 62134],   # p=7
], float)

fig, ax = plt.subplots(figsize=(7.5, 6))
Mm = np.ma.masked_where(M == 0, M)
im = ax.imshow(Mm, cmap="viridis", norm=LogNorm(vmin=1, vmax=M.max()), aspect="auto")
for i in range(len(ps)):
    for j in range(len(dfts)):
        if M[i, j]:
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < 5000 else "black", fontsize=9)
ax.set_xticks(range(len(dfts))); ax.set_xticklabels(dfts)
ax.set_yticks(range(len(ps))); ax.set_yticklabels(ps)
ax.set_xlabel("DFT-sparsity  →  nº de espirales REALES de rejilla (|w|=ρ)")
ax.set_ylabel("Prony p  →  nº de espirales ABSTRACTAS (w libre)")
ax.set_title("Mapa de complejidad — 13 reinas (73712 soluciones)\n"
             "esquina inf-izq vacía: 'simple en ambas' solo lo es el espiral puro (arriba-izq)")
# resaltar los dos extremos
ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="red", lw=2))
fig.colorbar(im, label="nº de soluciones (log)")
fig.tight_layout()
fig.savefig("complexity_map.png", dpi=130)
print("Guardado: complexity_map.png")


if __name__ == "__main__":
    pass
