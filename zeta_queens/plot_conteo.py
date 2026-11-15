# plot_conteo.py — cuanto se acerca el conteo predicho al real a medida que se conocen
# mas filas.  Datos de: ./zeta_big curves 13 > curves_13.csv ; ./zeta_big rms 8 14 > rms_8_14.csv
import numpy as np
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})

c = np.genfromtxt("curves_13.csv", delimiter=",", names=True)
x, real, trend = c["x"], c["real"], c["d0"]
pred = [(1, "1ª fila"), (2, "2 filas"), (3, "3 filas")]

fig = plt.figure(figsize=(12, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.22)

def residual_panel(ax, sel, title, labels):
    ax.plot(x[sel], (real - trend)[sel], color=AXIS, lw=3.2, solid_capstyle="round", zorder=1)
    for k, (d, name) in enumerate(pred):
        ax.plot(x[sel], (c[f"d{d}"] - trend)[sel], color=SLOT[k], lw=1.3, zorder=2 + k)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("solución leída como número (base 14), en unidades de la primera cifra")
    ax.set_ylabel("conteo − tendencia  r·P(x)")
    if labels:
        h = [plt.Line2D([], [], color=AXIS, lw=3.2, label="real (N = 13)")]
        h += [plt.Line2D([], [], color=SLOT[k], lw=1.6, label=f"predicho conociendo {n}") for k, (_, n) in enumerate(pred)]
        ax.legend(handles=h, frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)

ax1 = fig.add_subplot(gs[0, :])
residual_panel(ax1, slice(None), "N = 13 · lo que la tendencia no explica (gris) y lo que predice conocer las primeras filas", True)
ax1.set_xlim(x[0], x[-1])
ax1.text(0.99, 0.04, "Conocer la 1ª fila ya da el bulto del borde; 2 y 3 filas añaden el peine diagonal.",
         transform=ax1.transAxes, color=INK2, fontsize=9, ha="right")

ax2 = fig.add_subplot(gs[1, 0])
sel = (x >= 7) & (x < 8)
residual_panel(ax2, sel, "Zoom en la celda de la primera cifra = 7", True)
ax2.set_xlim(7, 8)

# error cuadratico medio frente a filas conocidas
ax3 = fig.add_subplot(gs[1, 1])
r = np.genfromtxt("rms_8_14.csv", delimiter=",", names=True)
for k, N in enumerate(range(8, 15)):
    m = (r["N"] == N) & (r["rms"] > 0)
    ax3.plot(r["d"][m], r["rms"][m], color=RAMP[k], lw=2, marker="o", ms=4.5,
             markeredgecolor=SURF, markeredgewidth=1, label=f"N = {N}")
    if N in (8, 11, 14):
        ax3.annotate(f"N={N}", (r["d"][m][0], r["rms"][m][0]), xytext=(-30, 0),
                     textcoords="offset points", color=INK2, fontsize=9, va="center")
ax3.set_yscale("log")
ax3.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), label="0.5: redondear da\nel conteo exacto")
ax3.axhline(1 / np.sqrt(12), color=MUTED, lw=1, ls=(0, (1, 2)), label="1/√12: 1 solución por\ncelda, solo falta posición")
ax3.set_xlim(-1.6, 14.4)
ax3.set_xticks(range(0, 15))
ax3.set_xlabel("filas conocidas d  (d = 0: solo la tendencia)")
ax3.set_ylabel("error RMS del conteo (soluciones)")
ax3.set_title("Error del conteo predicho según las filas que se conocen", loc="left", color=INK, fontsize=11)
ax3.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right", labelcolor=INK2, title="tamaño del tablero", title_fontsize=8.5)
ax3.set_ylim(0.15, 6e4)
ax3.text(0.0, -0.2, "Con d = N−1 el error es exactamente 0: la última reina queda forzada.",
         transform=ax3.transAxes, color=INK2, fontsize=8.5)

fig.savefig("conteo_vs_prediccion.png", dpi=160, bbox_inches="tight")
