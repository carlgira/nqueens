"""
La función del perfil: ¿hay una curva que pase por los <c_k>?

Sí, y sale de la máxima entropía. Los tres momentos invariantes que
encontramos (sum c_k, sum k*c_k, sum k^2*c_k) son restricciones lineales sobre
el vector de ocupación; la distribución de máxima entropía compatible con
restricciones lineales es una familia exponencial, que para variables binarias
da una Fermi-Dirac:

	<c_k> = 1 / (1 + exp(g*((k-N)^2 - r^2)))

Los ajustes dan g*N^2 y r/N CONSTANTES, así que en la variable reescalada
u = (k-N)/N la función no depende de N en absoluto:

	<c_k> = 1 / (1 + exp(g0*(u^2 - rho^2)))      u = (k-N)/N

con g0 ~ 5.26 y rho ~ 0.549 medidos sobre N = 10..14.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.cylinder import sol_spectra


def profile(n):
	S = sol_spectra(n)
	k = np.arange(1, 2 * n)
	return k, np.array([sum(kk in A for A, _ in S) for kk in k]) / len(S)


def fit(ns=(10, 11, 12, 13, 14)):
	"""Ajuste conjunto de g0 y rho sobre la variable reescalada."""
	U, Y = [], []
	for n in ns:
		k, p = profile(n)
		ok = (p > 0) & (p < 1)
		U.append(((k[ok] - n) / n) ** 2.0)
		Y.append(np.log(p[ok] / (1 - p[ok])))
	U, Y = np.concatenate(U), np.concatenate(Y)
	A = np.vstack([U, np.ones_like(U)]).T
	(a, b), *_ = np.linalg.lstsq(A, Y, rcond=None)
	return -a, np.sqrt(-b / a)					# g0, rho


def curve(u, g0, rho):
	return 1 / (1 + np.exp(g0 * (u ** 2 - rho ** 2)))


def plot(ns=(8, 10, 12, 14), path='diag_waves/fig_profile.png'):
	g0, rho = fit()
	fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
	cols = plt.cm.viridis(np.linspace(.15, .85, len(ns)))
	worst = 0.0
	uu = np.linspace(-1, 1, 400)
	for c, n in zip(cols, ns):
		k, p = profile(n)
		u = (k - n) / n
		ax[0].plot(k, p, 'o-', color=c, ms=4, lw=1, label=f'N={n}')
		ax[1].plot(u, p, 'o', color=c, ms=5, label=f'N={n}')
		worst = max(worst, np.abs(curve(u, g0, rho) - p).max())
	ax[1].plot(uu, curve(uu, g0, rho), 'k--', lw=2.2,
			   label=f'$1/(1+e^{{{g0:.2f}(u^2-{rho**2:.3f})}})$')
	ax[0].set_xlabel('diagonal $k$'); ax[0].set_ylabel('$\\langle c_k\\rangle$')
	ax[0].set_title('Los perfiles crudos')
	ax[1].set_xlabel('$u = (k-N)/N$'); ax[1].set_ylabel('$\\langle c_k\\rangle$')
	ax[1].set_title(f'Reescalado + la curva de máxima entropía\n'
					f'error máx = {worst:.3f}')
	for a in ax:
		a.grid(alpha=.25); a.legend(fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=150, bbox_inches='tight')
	print(f'g0 = {g0:.4f}   rho = {rho:.4f}   error maximo = {worst:.4f}')
	return path


if __name__ == '__main__':
	print(plot())
