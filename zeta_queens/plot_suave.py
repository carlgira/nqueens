# plot_suave.py — el primer 15 % de la curva con una sola solucion por orbita.
# Datos: ./zeta_suave 10 16 0.15 > suave_10_16.txt ; ./zeta_suave dump 14 0.15 > suave_14.csv
import re
import numpy as np
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT = ["primera", "azar", "tv", "huecos", "ritmo"]
NAME = {"primera": "la primera de cada órbita", "azar": "al azar", "tv": "menos saltos (TV)",
        "huecos": "huecos parejos", "ritmo": "ritmo constante"}
COL = dict(zip(CRIT, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})

rows = [l.strip().split(",") for l in open("suave_14.csv")][1:]
data = {c: ([], []) for c in CRIT + ["entropia", "todas"]}
for c, u, i in rows:
    data[c][0].append(float(u)); data[c][1].append(int(i))
data = {c: (np.array(a), np.array(b)) for c, (a, b) in data.items()}
w = max(b.max() for _, b in data.values()) + 1

fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.8), gridspec_kw=dict(hspace=0.36, wspace=0.24))

ax = axs[0, 0]
grid = np.arange(w)
for c in CRIT:
    idx = data[c][1]
    cnt = np.searchsorted(idx, grid, side="right")
    dev = cnt * (w / len(idx)) - (grid + 1)
    ax.plot(grid / w, 100 * dev / w, color=COL[c], lw=1.8, label=NAME[c])
ax.axhline(0, color=AXIS, lw=0.8)
ax.set_xlabel("posición dentro del tramo")
ax.set_ylabel("desviación frente a la curva real (% del tramo)")
ax.set_title("N = 14 · cuánto se aparta la curva adelgazada", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="best", labelcolor=INK2)

ax = axs[0, 1]
edges = np.arange(1, 9)
width = 0.2
show = ["primera", "azar", "huecos", "ritmo"]
for j, c in enumerate(show):
    g = np.diff(data[c][1])
    h = [np.mean(g == e) for e in edges[:-1]] + [np.mean(g >= edges[-1])]
    ax.bar(edges + (j - 1.5) * width, h, width=width * 0.9, color=COL[c], label=NAME[c])
ax.set_xticks(edges); ax.set_xticklabels([str(e) for e in edges[:-1]] + ["≥8"])
ax.set_xlabel("soluciones avanzadas entre dos puntos elegidos")
ax.set_ylabel("fracción de huecos")
ax.set_title("N = 14 · tamaño de los huecos", loc="left", color=INK, fontsize=11)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)

ax = axs[1, 0]
lo, hi = 2.40, 2.56
rowsplot = [("todas", None)] + [(NAME[c], c) for c in ("primera", "azar", "huecos", "ritmo")]
labels = []
for k, (label, c) in enumerate(rowsplot):
    uu = data["todas" if c is None else c][0]
    uu = uu[(uu >= lo) & (uu < hi)]
    ax.vlines(uu, k + 0.12, k + 0.88, color=INK2 if c is None else COL[c], lw=0.5)
    labels.append(f"{label}\n({len(uu)} puntos)")
ax.set_xlim(lo, hi); ax.set_ylim(0, len(rowsplot))
ax.set_yticks(np.arange(len(rowsplot)) + 0.5); ax.set_yticklabels(labels, fontsize=8, color=INK2)
ax.tick_params(axis="y", length=0)
ax.grid(False)
ax.set_xlabel("solución leída como número (base 15)")
ax.set_title("N = 14 · zoom: dónde quedan los puntos", loc="left", color=INK, fontsize=11)

ax = axs[1, 1]
Ns, err = [], {c: [] for c in CRIT}
tramo = None
for line in open("suave_10_16.txt"):
    m = re.match(r"N=(\d+) Q=\d+ tramo=(\d+)", line)
    if m: Ns.append(int(m.group(1))); tramo = int(m.group(2))
    m = re.match(r"\s+(\w+)\s+en u:.*error_curva=([\d.]+)", line)
    if m and m.group(1) in err: err[m.group(1)].append(100 * float(m.group(2)) / tramo)
for c in CRIT:
    ax.plot(Ns, err[c], color=COL[c], lw=2, marker="o", ms=5, markeredgecolor=SURF, label=NAME[c])
ax.set_xticks(Ns); ax.set_ylim(0, None)
ax.set_xlabel("N")
ax.set_ylabel("error RMS de la curva (% del tramo)")
ax.set_title("Error frente a la curva real, según N", loc="left", color=INK, fontsize=11)

fig.savefig("suave_tramo.png", dpi=160, bbox_inches="tight")
