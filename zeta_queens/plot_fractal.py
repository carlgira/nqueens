# plot_fractal.py — las soluciones como fractal de prefijos: cada nivel es una cifra mas.
# Datos: ./zeta_fractal 14 > sols_14.txt ; ./zeta_huecos 10 16 > huecos_10_16.csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("azul", [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})

N = 14
B = N + 1
rows = open("sols_14.txt").read().split()
D = np.array([[int(ch, 16) + 1 for ch in r] for r in rows], dtype=np.int64)   # cifras 1..N
Q = len(D)

def icicle(ax, lo, hi, levels, px=2200):
    """Cada fila es un nivel; cada celda ocupada se pinta segun sus soluciones (log)."""
    img = np.zeros((levels, px))
    w = np.array([float(B) ** -k for k in range(N)])
    for d in range(1, levels + 1):
        start = D[:, :d] @ w[:d]                # inicio de la celda de nivel d
        cellw = w[d - 1]
        keep = (start + cellw > lo) & (start < hi)
        st, cnt = np.unique(np.round(start[keep], 10), return_counts=True)
        a = np.clip(((st - lo) / (hi - lo) * px).astype(int), 0, px - 1)
        b = np.clip(((st + cellw - lo) / (hi - lo) * px).astype(int), 0, px - 1)
        v = np.log1p(cnt)
        v = v / v.max()
        for i, j, c in zip(a, b, v):
            img[d - 1, i:max(i + 1, j)] = np.maximum(img[d - 1, i:max(i + 1, j)], 0.15 + 0.85 * c)
    ax.imshow(img, aspect="auto", cmap=CMAP, vmin=0, vmax=1, extent=[lo, hi, levels + 0.5, 0.5], interpolation="nearest")
    ax.set_yticks(range(1, levels + 1))
    ax.set_ylabel("nivel (cifras conocidas)")
    ax.spines["left"].set_visible(False)

fig, axs = plt.subplots(2, 2, figsize=(13, 9.6), gridspec_kw=dict(hspace=0.33, wspace=0.2, width_ratios=[1.45, 1]))

ax = axs[0, 0]
icicle(ax, 1, B, 7)
ax.set_xlabel("solución leída como número (base 15)")
ax.set_title("N = 14 · cada nivel subdivide al anterior (blanco = sin soluciones)", loc="left", color=INK, fontsize=11)

ax = axs[1, 0]
lo = 2 + 7 / B
hi = lo + 1 / B
icicle(ax, lo, hi, 8)
ax.set_xlabel("zoom: solo las que empiezan por 2, 7")
ax.set_title("El mismo dibujo dentro de una celda", loc="left", color=INK, fontsize=11)

# tapiz: cifras impares en x, pares en y (cada par de cifras elige una casilla dentro de otra)
ax = axs[0, 1]
wx = np.array([float(B) ** -k for k in range(N // 2)])
x = D[:, 0::2][:, :N // 2] @ wx
y = D[:, 1::2][:, :N // 2] @ wx
H, _, _ = np.histogram2d(y, x, bins=700, range=[[1, B], [1, B]])
ax.imshow(np.log1p(H), origin="lower", cmap=CMAP, extent=[1, B, 1, B], interpolation="nearest")
ax.set_xlabel("cifras 1, 3, 5… (primera = parte entera)")
ax.set_ylabel("cifras 2, 4, 6…")
ax.set_title("Tapiz: las cifras repartidas en dos ejes", loc="left", color=INK, fontsize=11)

ax = axs[1, 1]
r = np.genfromtxt("huecos_10_16.csv", delimiter=",", names=True)
for k, n in enumerate(range(10, 17)):
    m = r["N"] == n
    occ = r["ocupados"][m]
    ratio = occ[1:] / occ[:-1]
    dim = np.log(np.maximum(ratio, 1)) / np.log(n + 1)
    ax.plot(r["d"][m][1:], dim, color=RAMP[k], lw=2, marker="o", ms=3.5, markeredgecolor=SURF, markeredgewidth=0.6, label=f"N = {n}")
ax.set_xlabel("nivel d")
ax.set_ylabel("dimensión local: log(subceldas ocupadas) / log(N+1)")
ax.set_ylim(0, 1)
ax.grid(True, color=GRID, lw=0.6)
ax.set_title("Cuánto se llena cada nivel", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK2)

fig.savefig("fractal_soluciones.png", dpi=160, bbox_inches="tight")
