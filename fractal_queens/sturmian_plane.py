"""
Figura 2: el analogo de Mandelbrot.
Plano de parametros (alpha, beta) -> colocacion c_k = rango de frac(alpha*k+beta).
Color = numero de conflictos. Verde = solucion exacta.
La estructura es celular, delimitada por fracciones de Farey, y autosemejante
bajo el mapa de Gauss alpha -> 1/alpha mod 1.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def conflict_map(N, a0, a1, b0, b1, res, chunk=40000):
	A, B = np.meshgrid(np.linspace(a0, a1, res), np.linspace(b0, b1, res))
	al = A.ravel(); be = B.ravel()
	out = np.empty(al.size, np.int16)
	k = np.arange(N); i = np.arange(N)
	for s in range(0, al.size, chunk):
		a = al[s:s + chunk, None]; b = be[s:s + chunk, None]
		x = (a * k + b) % 1.0
		r = np.argsort(np.argsort(x, axis=1), axis=1)          # rango = columna
		d = 0
		for v in (i + r, i - r):
			v = np.sort(v, axis=1)
			d += N - (1 + (np.diff(v, axis=1) != 0).sum(1))     # colisiones
		out[s:s + chunk] = d
	return out.reshape(res, res)


if __name__ == '__main__':
	N, res = 13, 900
	views = [(0.0, 1.0, 0.0, 1.0, "todo el plano"),
	         (0.30, 0.40, 0.0, 1.0, "zoom x10"),
	         (0.3805, 0.3855, 0.0, 1.0, "zoom x200")]

	fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), facecolor='#0d0d12')
	for a, (a0, a1, b0, b1, t) in zip(ax, views):
		M = conflict_map(N, a0, a1, b0, b1, res)
		a.imshow(M, origin='lower', extent=[a0, a1, b0, b1], aspect='auto',
		         cmap='twilight_shifted', interpolation='nearest')
		sol = M == 0
		if sol.any():
			yy, xx = np.nonzero(sol)
			a.scatter(a0 + xx * (a1 - a0) / (res - 1), b0 + yy * (b1 - b0) / (res - 1),
			          s=1.2, c='#00ff9c', linewidths=0)
		a.set_title(f"{t}   ({int(sol.sum())} px exactos)", color='#e8e8f0', fontsize=11)
		a.tick_params(colors='#8a8a99')
	fig.suptitle(f"N={N}: conflictos de c_k = rango de frac(alpha*k+beta)  -  verde = solucion",
	             color='#e8e8f0', fontsize=13)
	fig.tight_layout()
	fig.savefig('/Users/carlgira/files/nqueens/nqueens/fractal_queens/fig_sturmian_plane.png',
	            dpi=130, facecolor='#0d0d12')
	print('fig_sturmian_plane.png')
