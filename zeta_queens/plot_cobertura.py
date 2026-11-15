# plot_cobertura.py — parte de Q vista por orbitas frente al tramo de curva recorrido,
# contra la ley 1-(1-2f)^4, y estimacion de Q sin conocerlo.  Datos: ./zeta_cob > cob_8_19.csv
import numpy as np
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
})

r = np.genfromtxt("cob_8_19.csv", delimiter=",", names=True)

def curves(N):
    m = r["N"] == N
    Q = r["Q"][m][0]
    return Q, np.cumsum(r["n_c1"][m]), np.cumsum(r["h_m"][m])

def qhat(W, S):
    # S/W = g(f) con S/Q = 1-(1-2f)^4; se resuelve f por biseccion en (0, 1/2)
    g = lambda f: (1 - (1 - 2 * f) ** 4) / f
    lo, hi = 1e-9, 0.5
    target = S / W
    if target >= g(lo) or target <= g(hi):
        return np.nan
    for _ in range(80):
        mid = (lo + hi) / 2
        if g(mid) > target: lo = mid
        else: hi = mid
    return W / lo

print("N  Q  |  error de Q estimado (%) con k=2 .. 6, y tramo recorrido")
est = {}
for N in range(11, 20):
    Q, W, S = curves(N)
    row = []
    for k in range(2, 7):
        q = qhat(W[k - 1], S[k - 1])
        row.append((k, 100 * (q / Q - 1), 100 * W[k - 1] / Q))
    est[N] = row
    print(N, int(Q), " ".join(f"k{k}:{e:+.1f}%({t:.0f}%)" for k, e, t in row))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw=dict(wspace=0.27))

ax = axs[0]
for i, N in enumerate(range(11, 20)):
    Q, W, S = curves(N)
    ax.plot(np.r_[0, W / Q], np.r_[0, S / Q], color=RAMP[i], lw=1.8, marker="o", ms=3,
            markeredgecolor=SURF, markeredgewidth=0.5, label=f"N = {N}")
f = np.linspace(0, 0.5, 200)
ax.plot(f, 1 - (1 - 2 * f) ** 4, color=ORANGE, lw=1.6, ls=(0, (4, 3)), label="1 − (1 − 2f)⁴")
ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02)
ax.set_xlabel("tramo recorrido de la curva, f")
ax.set_ylabel("parte de Q vista por órbitas")
ax.set_title("El porcentaje por tramo sí se mantiene", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right", labelcolor=INK2)

ax = axs[1]
Ns = np.arange(11, 20)
for pos, (ff, col, name) in enumerate([(0.12, BLUE, "tramo fijo 12 %"), (0.20, AQUA, "tramo fijo 20 %")]):
    y = [np.interp(ff, np.r_[0, curves(N)[1] / curves(N)[0]], np.r_[0, curves(N)[2] / curves(N)[0]]) for N in Ns]
    ax.plot(Ns, y, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURF, label=name)
y = [curves(N)[2][2] / curves(N)[0] for N in Ns]
ax.plot(Ns, y, color=ORANGE, lw=2, marker="s", ms=5, markeredgecolor=SURF, label="cifra fija: primera cifra ≤ 3")
ax.set_ylim(0.45, 1)
ax.set_xticks(Ns)
ax.set_xlabel("N")
ax.set_ylabel("parte de Q vista")
ax.set_title("Lo que cae con N es la cifra fija", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)

ax = axs[2]
for j, N in enumerate(Ns):
    W, Q = curves(N)[1], curves(N)[0]
    ax.bar(N, W[0] / Q * N, width=0.62, color=BLUE if N % 2 else "#86b6ef")
ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
ax.set_xticks(Ns)
ax.set_ylim(0, 0.6)
ax.set_xlabel("N (oscuro = impar, claro = par)")
ax.set_ylabel("N · parte de soluciones con reina en la esquina")
ax.set_title("Reina en la esquina: ≈ 0.47 / N", loc="left", color=INK, fontsize=11)

fig.savefig("cobertura_N.png", dpi=160, bbox_inches="tight")
