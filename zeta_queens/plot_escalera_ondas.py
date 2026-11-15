# plot_escalera_ondas.py — conteo acumulado de soluciones (escalera real) frente a lo
# predicho por el experimento inicial: tendencia + K ondas buscadas.
# Datos: ./zeta_big wavecurves 8 > wavecurves_8.csv ; ./zeta_big wavecurves 13 > wavecurves_13.csv
import numpy as np
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PRED = [("trend", "tendencia sola"), ("w1", "1 onda (4 parámetros)"), ("w4", "4 ondas (13)"),
        ("w12", "12 ondas (37)"), ("w40", "40 ondas (121)")]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})

def panel(ax, c, sel, title, xlabel, legend):
    x = c["x"][sel]
    ax.step(x, c["real"][sel] + 0.5, where="post", color=INK, lw=1.4, label="conteo real", zorder=6)
    for k, (col, name) in enumerate(PRED):
        ax.plot(x, c[col][sel], color=SLOT[k], lw=1.8, label=name, zorder=5 - k * 0.1)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("soluciones hasta ese número")
    if legend:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)

c8 = np.genfromtxt("wavecurves_8.csv", delimiter=",", names=True)
c13 = np.genfromtxt("wavecurves_13.csv", delimiter=",", names=True)

fig, axs = plt.subplots(2, 2, figsize=(12, 9.5), gridspec_kw=dict(hspace=0.38, wspace=0.2))

panel(axs[0, 0], c8, slice(None), "N = 8 · las 92 soluciones", "solución leída como número (base 9, primera cifra = parte entera)", True)
axs[0, 0].text(0.98, 0.04, "ojo: 40 ondas son 121 parámetros\npara 92 soluciones (sobreajuste)",
               transform=axs[0, 0].transAxes, ha="right", color=INK2, fontsize=8.5)

sel = (c8["x"] >= 4) & (c8["x"] < 6)
panel(axs[0, 1], c8, sel, "N = 8 · zoom: soluciones que empiezan por 4 o 5", "solución leída como número (base 9)", False)

panel(axs[1, 0], c13, slice(None), "N = 13 · las 73 712 soluciones", "solución leída como número (base 14)", True)

sel = (c13["x"] >= 7.40) & (c13["x"] < 7.43)
panel(axs[1, 1], c13, sel, "N = 13 · zoom en una ventana de 0.03 (603 soluciones)\n", "solución leída como número (base 14)", False)
axs[1, 1].text(0.0, 1.0, "error máximo aquí: tendencia 477 · K=1 757 · K=4 501 · K=12 146 · K=40 79",
               transform=axs[1, 1].transAxes, va="bottom", color=INK2, fontsize=8.5)

fig.savefig("escalera_ondas.png", dpi=160, bbox_inches="tight")
