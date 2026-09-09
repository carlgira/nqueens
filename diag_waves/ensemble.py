"""Ensemble statistics of the solution spectra (the 'diffraction' of all solutions)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sol_coefs(n):
	out, cur, full = [], [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if r == n:
			c = np.zeros(2 * n)
			for i, s in enumerate(cur):
				c[i + s + 1] += 1
			out.append(c); return
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b; k = b.bit_length() - 1
			cur.append(k); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	return np.array(out)


def plot(ns=(8, 10, 12, 14), path='diag_waves/fig_ensemble.png'):
	fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
	cols = plt.cm.viridis(np.linspace(.15, .85, len(ns)))
	for c, n in zip(cols, ns):
		C = sol_coefs(n)
		mu = C.mean(0)[1:2 * n]
		k = np.arange(1, 2 * n)
		ax[0].plot(k, mu, 'o-', color=c, ms=4, label=f'N={n} ({len(C)} sols)')
		ax[1].plot(k / (2 * n), mu, 'o-', color=c, ms=4, label=f'N={n}')
		ev = np.linalg.eigvalsh(np.cov(C[:, 1:2 * n].T))[::-1]
		print(f'N={n}: <c_k> {mu.min():.3f}..{mu.max():.3f}  '
			  f'3 modos PCA = {100*ev[:3].sum()/ev.sum():.0f}% de la varianza')
	ax[0].axhline(.5, color='#adb5bd', ls=':', lw=1.2)
	ax[0].set_xlabel('diagonal $k$'); ax[0].set_ylabel('$\\langle c_k \\rangle$')
	ax[0].set_title('Ocupación media de cada diagonal sobre TODAS las soluciones')
	ax[0].legend(fontsize=9)
	ax[1].set_xlabel('$k / 2N$'); ax[1].set_ylabel('$\\langle c_k \\rangle$')
	ax[1].set_title('Reescalado: el perfil colapsa a una curva universal')
	ax[1].legend(fontsize=9)
	for a in ax: a.grid(alpha=.25)
	fig.tight_layout()
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
