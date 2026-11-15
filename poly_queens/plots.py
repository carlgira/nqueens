"""Los ceros de C_N(z) cerrandose sobre z=1, y la ley R ~ 4.9/N."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .conflict_poly import conflict_hist, roots

HERE = os.path.dirname(__file__)


def hist(n):
	f = os.path.join(HERE, f"hist_{n}.npy")
	if os.path.exists(f):
		return np.load(f)
	h = conflict_hist(n)
	np.save(f, h)
	return h


def main(ns=(8, 10, 12)):
	fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))

	for n, col in zip(ns, ("#9ecae1", "#4292c6", "#08306b")):
		r = roots(hist(n))
		R = np.abs(r - 1).min()
		ax[0].scatter(r.real, r.imag, s=16, color=col, label=f"N={n}  (R={R:.3f})",
		              zorder=3, edgecolors="none")
		th = np.linspace(0, 2 * np.pi, 400)
		ax[0].plot(1 + R * np.cos(th), R * np.sin(th), color=col, lw=1.1, ls="--", alpha=.8)

	ax[0].scatter([0, 1], [0, 0], marker="x", s=90, color="crimson", zorder=5, lw=2)
	ax[0].annotate("z=0\nQ(N)", (0, 0), textcoords="offset points", xytext=(-6, 12),
	               ha="center", color="crimson", fontsize=9)
	ax[0].annotate("z=1\nN!", (1, 0), textcoords="offset points", xytext=(8, 12),
	               ha="center", color="crimson", fontsize=9)
	ax[0].axhline(0, color="k", lw=.5, alpha=.3)
	ax[0].axvline(0, color="k", lw=.5, alpha=.3)
	ax[0].set_title("ceros de $C_N(z)$ y disco de convergencia en $z=1$")
	ax[0].set_xlabel("Re z"); ax[0].set_ylabel("Im z")
	ax[0].legend(fontsize=8, loc="upper left")
	ax[0].set_aspect("equal"); ax[0].set_xlim(-3.2, 2.2); ax[0].set_ylim(-2.4, 2.4)

	NS = list(range(5, 13))
	Rs = [np.abs(roots(hist(n)) - 1).min() for n in NS]
	ax[1].plot(NS, Rs, "o-", color="#08306b", label="R(N) = min |r - 1|")
	ax[1].plot(NS, [4.9 / n for n in NS], "--", color="crimson", label="4.9 / N")
	ax[1].axhline(1, color="k", lw=1, alpha=.5)
	ax[1].annotate("R=1: por debajo, z=0 queda fuera\ny la serie de cumulos diverge",
	               (8.6, 1.02), fontsize=8.5, color="k")
	ax[1].set_xlabel("N"); ax[1].set_ylabel("radio de convergencia")
	ax[1].set_title("el disco se cierra como $4.9/N$")
	ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)

	fig.tight_layout()
	out = os.path.join(HERE, "fig_zeros.png")
	fig.savefig(out, dpi=140)
	print("->", out)
	print(" N   R      N*R")
	for n, R in zip(NS, Rs):
		print(f"{n:2d} {R:.4f} {n*R:7.3f}")


if __name__ == "__main__":
	main()
