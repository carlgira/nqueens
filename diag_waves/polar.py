"""
La curva polar de una solución.

Los c_k de la banda d+ son M = 2N-1 valores 0/1 colocados en ángulos
theta_k = 2*pi*k/M. La curva que pasa EXACTAMENTE por los M puntos y es
suave entre ellos es el interpolante trigonométrico:

	r(theta) = (1/M) * sum_m V_m * exp(i*m*theta),      V = DFT(c)

M es impar (2N-1), así que no hay término de Nyquist y r sale real sin arreglos.
V_0 = N siempre (hay N reinas). Los demás V_m son la transformada del conjunto
de diagonales ocupadas -- literalmente su patrón de difracción.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import channels, shift, conflicts


def occupancy_ring(perm, band='+'):
	"""Vector 0/1 de longitud M = 2N-1 con las diagonales ocupadas."""
	n = len(perm)
	a, b = shift(*channels(perm), n)
	v = np.zeros(2 * n - 1)
	for k in (a if band == '+' else b):
		v[k - 1] += 1
	return v


def polar_curve(v, pts=1200):
	"""r(theta) interpolando v en los M ángulos equiespaciados."""
	m = len(v)
	V = np.fft.fft(v)
	th = np.linspace(0, 2 * np.pi, pts)
	freqs = np.fft.fftfreq(m, 1 / m)
	r = (V[None, :] * np.exp(1j * np.outer(th, freqs))).sum(1).real / m
	return th, r, V


def plot(n=9, perms=None, path='diag_waves/fig_polar.png'):
	if perms is None:
		perms = [([0, 2, 5, 7, 1, 3, 8, 6, 4], 'solución A'),
				 ([0, 3, 6, 8, 1, 4, 7, 5, 2], 'solución B'),
				 ([0, 1, 3, 6, 2, 8, 4, 7, 5], 'NO solución')]
	m = 2 * n - 1
	fig, ax = plt.subplots(2, len(perms), figsize=(5.4 * len(perms), 9.6),
						   subplot_kw={}, squeeze=False)
	for j, (p, name) in enumerate(perms):
		v = occupancy_ring(p)
		th, r, V = polar_curve(v)
		dp, dm = channels(p)
		nc = conflicts(dp)
		col = '#e8590c' if nc == 0 else '#495057'

		a = fig.add_subplot(2, len(perms), j + 1, projection='polar')
		ax[0, j].remove()
		a.plot(th, r, color=col, lw=2)
		tk = 2 * np.pi * np.arange(m) / m
		a.scatter(tk[v > 0], v[v > 0], color=col, s=70, zorder=4,
				  edgecolor='k', linewidth=.6)
		a.scatter(tk[v == 0], np.zeros((v == 0).sum()), color='#ced4da', s=40,
				  zorder=4, edgecolor='k', linewidth=.4)
		a.set_ylim(r.min() - .15, max(1.05, r.max() + .05))
		a.set_title(f'{name}\n{p}\nconflictos $d^+$ = {nc}', fontsize=10, pad=16, color=col)
		a.set_xticks(tk[::2])
		a.set_xticklabels([str(k + 1) for k in range(0, m, 2)], fontsize=7)
		a.set_yticklabels([])

		b = ax[1, j]
		mm = np.arange(m)
		b.stem(mm, np.abs(V), linefmt=col, markerfmt='o', basefmt=' ')
		b.axhline(np.sqrt(n), color='#1c7ed6', ls='--', lw=1.4,
				  label=f'$\\sqrt{{N}}$ = {np.sqrt(n):.2f}')
		b.set_xlabel('modo $m$'); b.set_ylabel('$|V_m|$')
		b.set_title(f'$V_0$ = {abs(V[0]):.0f}   ·   '
					f'$\\langle |V_m|^2\\rangle_{{m\\neq0}}$ = '
					f'{(np.abs(V[1:])**2).mean():.2f}', fontsize=10)
		b.legend(fontsize=8)
		b.set_ylim(0, max(4, np.abs(V[1:]).max() * 1.25))

	fig.suptitle(f'N={n}: la curva polar que interpola los coeficientes de la banda $d^+$',
				 fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .95))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
