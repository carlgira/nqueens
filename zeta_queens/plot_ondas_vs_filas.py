# plot_ondas_vs_filas.py — error del conteo frente al numero de parametros:
# experimento inicial (tendencia + K ondas buscadas) contra el modelo por filas conocidas.
# Datos: ./zeta_big waves 8 13 > waves_8_13.csv ; ./zeta_big rms 8 14 > rms_8_14.csv ; multi_10_14.txt
import re
import numpy as np
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})

w = np.genfromtxt("waves_8_13.csv", delimiter=",", names=True)
r = np.genfromtxt("rms_8_14.csv", delimiter=",", names=True)

# celdas no vacias por profundidad: columna "celdas" del nivel k = d+1 en multi_10_14.txt
cells, n = {}, None
for line in open("multi_10_14.txt"):
    m = re.match(r"==== N=(\d+)", line)
    if m: n = int(m.group(1)); cells[n] = {0: 1}
    m = re.match(r"\s+(\d+)\s+(\d+)\s", line)
    if m and n: cells[n][int(m.group(1)) - 1] = int(m.group(2))

fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), gridspec_kw=dict(hspace=0.35, wspace=0.22))
for ax, N in zip(axs.flat, (10, 11, 12, 13)):
    m = w["N"] == N
    ax.plot(w["param"][m], w["rms"][m], color=BLUE, lw=2, label="ondas buscadas (K = 0…40)")
    for K in (1, 4, 12, 40):
        i = np.where(m & (w["K"] == K))[0][0]
        ax.plot(w["param"][i], w["rms"][i], "o", ms=6, color=BLUE, markeredgecolor=SURF, markeredgewidth=1.2)
        ax.annotate(f"K={K}", (w["param"][i], w["rms"][i]), xytext=(-5, -13), textcoords="offset points", color=INK2, fontsize=8, ha="right")
    ds = [d for d in sorted(cells[N]) if d <= 4]
    pr = [cells[N][d] for d in ds]
    er = [r["rms"][(r["N"] == N) & (r["d"] == d)][0] for d in ds]
    ax.plot(pr, er, color=ORANGE, lw=2, marker="s", ms=6, markeredgecolor=SURF, markeredgewidth=1.2,
            label="filas conocidas (d = 0…4)")
    for d, x_, y_ in zip(ds, pr, er):
        ax.annotate(f"d={d}", (x_, y_), xytext=(6, 4), textcoords="offset points", color=INK2, fontsize=8)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"N = {N}", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("parámetros (ondas: 3 por onda + 1 · filas: celdas no vacías)")
    ax.set_ylabel("error RMS del conteo (soluciones)")
    if N == 10:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
        ax.text(1.1, 0.53, "0.5: conteo exacto al redondear", color=MUTED, fontsize=8)

fig.suptitle("Con el mismo número de parámetros, las ondas buscadas se equivocan menos que las celdas por filas",
             x=0.07, ha="left", color=INK, fontsize=12)
fig.savefig("ondas_vs_filas.png", dpi=160, bbox_inches="tight")
