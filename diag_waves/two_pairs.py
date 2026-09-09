"""Los dos pares de dos: planos de la hélice (redundantes) vs bandas (distintas)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import band_series, channels, shift

C_X, C_Y, C_P, C_M = '#1c7ed6', '#7048e8', '#e8590c', '#0ca678'


def plot(n=9, p=(0, 2, 5, 7, 1, 3, 8, 6, 4), path='diag_waves/fig_two_pairs.png'):
	p = list(p)
	c = band_series(p); K = np.arange(len(c)); M = 2 * n - 1
	x = np.linspace(0, 2 * np.pi, 2000)
	X = c @ np.cos(np.outer(K, x))
	Y = c @ np.sin(np.outer(K, x))
	a, b = shift(*channels(p), n)

	fig = plt.figure(figsize=(15, 9))

	for j, (sig, col, lab) in enumerate(((X, C_X, 'X(x) = Re Z   plano horizontal'),
										 (Y, C_Y, 'Y(x) = Im Z   plano vertical'))):
		ax = fig.add_subplot(2, 2, j + 1)
		ax.plot(x, sig, color=col, lw=1.2)
		ax.set_xlim(0, 2 * np.pi); ax.set_xlabel('x'); ax.set_ylabel('amplitud')
		ax.set_title(f'{lab}\nsus coeficientes de Fourier: {[int(v) for v in c[1:14]]}…',
					 fontsize=10)

	th = 2 * np.pi * np.arange(M) / M
	thc = np.linspace(0, 2 * np.pi, 800)
	for j, (band, col, lab) in enumerate(((a, C_P, 'flor $d^+$'), (b, C_M, 'flor $d^-$'))):
		v = np.zeros(M); v[np.asarray(band) - 1] = 1
		V = np.fft.fft(v); fr = np.fft.fftfreq(M, 1 / M)
		r = (V[None] * np.exp(1j * np.outer(thc, fr))).sum(1).real / M
		q = fig.add_subplot(2, 2, 3 + j, projection='polar')
		q.plot(thc, r, color=col, lw=2)
		q.scatter(th[v > 0], v[v > 0], color=col, s=70, zorder=4, edgecolor='k', lw=.6)
		q.scatter(th[v == 0], v[v == 0], color='#ced4da', s=35, zorder=4)
		q.set_title(f'{lab}\ndiagonales {sorted(int(t) for t in band)}', fontsize=10, pad=14)
		q.set_yticklabels([]); q.set_xticklabels([])

	fig.text(.5, .53, 'ARRIBA: los dos planos de la hélice — MISMOS coeficientes, redundantes',
			 ha='center', fontsize=12, color='#c92a2a')
	fig.text(.5, .045, 'ABAJO: las dos bandas diagonales — flores DISTINTAS, información real',
			 ha='center', fontsize=12, color='#2b8a3e')
	fig.suptitle(f'N={n}, solución {p}: los dos pares de dos', fontsize=14)
	fig.tight_layout(rect=(0, .06, 1, .95))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
