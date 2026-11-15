# plot_simetrias.py — a que punto de la curva de conteo va cada solucion bajo las 7
# simetrias no triviales, y cuanta Q se recupera recorriendo solo un tramo de la curva.
# Datos: ./zeta_sim dump 12 > sim_12.csv ; ./zeta_sim cov 8 15 > cov_8_15.csv
import numpy as np
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

s = np.genfromtxt("sim_12.csv", delimiter=",", names=True)
Q = len(s)
pos = s["i"] / Q
PANELS = [("V", "V · espejo de columnas"), ("H", "H · filas al revés"), ("R180", "R180 = V·H"),
          ("T", "T · traspuesta"), ("R90", "R90 = V·T"), ("R270", "R270 = H·T"), ("A", "A · antitraspuesta")]

# limites de cada primera cifra en la curva de N = 12 (del conteo acumulado por cifra)
c = np.genfromtxt("cov_8_15.csv", delimiter=",", names=True)
edges = c["frac_curva"][c["N"] == 12][:-1]

fig, axs = plt.subplots(2, 4, figsize=(14, 7.6), gridspec_kw=dict(hspace=0.34, wspace=0.26))
for ax, (g, name) in zip(axs.flat, PANELS):
    for e in edges:
        ax.axvline(e, color=GRID, lw=0.6); ax.axhline(e, color=GRID, lw=0.6)
    ax.scatter(pos, s[g] / Q, s=1.0, color=BLUE, alpha=0.3, linewidths=0, rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(name, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel("posición de la solución en la curva")
    ax.set_ylabel("posición de su imagen")

ax = axs.flat[7]
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (1, 2)), label="sin simetrías")
ax.plot([0, 0.5, 1], [0, 1, 1], color=INK2, lw=1, ls=(0, (4, 3)), label="solo el espejo")
for k, N in enumerate(range(10, 16)):
    m = c["N"] == N
    ax.plot(np.r_[0, c["frac_curva"][m]], np.r_[0, c["frac_Q"][m]], color=RAMP[k], lw=2, marker="o", ms=3.5,
            markeredgecolor=SURF, markeredgewidth=0.7, label=f"N = {N}")
ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.02)
ax.grid(True, color=GRID, lw=0.6)
ax.set_title("Parte de Q que se recupera", loc="left", color=INK, fontsize=10.5)
ax.set_xlabel("tramo recorrido de la curva")
ax.set_ylabel("fracción de Q vista por órbitas")
ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK2)

fig.suptitle("N = 12 · a dónde manda cada simetría a las 14 200 soluciones (rejilla: primera cifra)",
             x=0.07, y=0.96, ha="left", color=INK, fontsize=12)
fig.savefig("simetrias_curva.png", dpi=160, bbox_inches="tight")
