"""
Las dos codificaciones, cara a cara.

  DIAGONAL:  frecuencia = cada diagonal por separado -> detecta conflictos, NO inyectiva
  EMPAREJADA: frecuencia = el par (d+, d-)           -> INYECTIVA, no detecta nada global

La informacion de la emparejada no desaparece: se esconde en cortes concretos.
En x = 2*pi*k/(2N) el termino de d+ da vueltas enteras y se anula, y queda solo d-.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import permutations

from diag_waves.diag_waves import band_series, channels, shift, conflicts

C_GOOD, C_BAD = '#e8590c', '#adb5bd'


def diag_wave(p, n, x):
	return band_series(p) @ np.exp(1j * np.outer(np.arange(4 * n), x))


def paired_wave(p, n, x):
	a, b = shift(*channels(p), n)
	f = np.asarray(a) * (2 * n) + np.asarray(b)
	return np.exp(1j * np.outer(f, x)).sum(0)


def plot(n=7, path='diag_waves/fig_paired.png'):
	M = (2 * n) ** 2
	x = 2 * np.pi * np.arange(M) / M
	S, O = {'d': [], 'p': []}, {'d': [], 'p': []}
	for p in permutations(range(n)):
		c = conflicts(channels(p)[0]) + conflicts(channels(p)[1])
		tgt = S if c == 0 else O
		tgt['d'].append(diag_wave(p, n, x)); tgt['p'].append(paired_wave(p, n, x))
	for k in ('d', 'p'):
		S[k] = np.array(S[k]); O[k] = np.array(O[k])

	fig = plt.figure(figsize=(16.5, 9.5))
	titles = {'d': 'DIAGONAL (la de siempre)', 'p': 'EMPAREJADA (inyectiva)'}
	for j, k in enumerate(('d', 'p')):
		a = fig.add_subplot(2, 2, j + 1, projection='polar')
		a.add_collection(LineCollection(
			[np.column_stack([x, r]) for r in np.abs(O[k])[::4]],
			colors=C_BAD, linewidths=.3, alpha=.05))
		for r in np.abs(S[k]):
			a.plot(x, r, color=C_GOOD, lw=.7, alpha=.75, zorder=3)
		a.set_ylim(0, n + .5); a.set_yticklabels([])
		a.set_title(f'{titles[k]}\n$|W(x)|$ · naranja = {len(S[k])} soluciones',
					fontsize=11, pad=16)

		b = fig.add_subplot(2, 2, j + 3)
		es = (np.abs(S[k]) ** 2).mean(1); eo = (np.abs(O[k]) ** 2).mean(1)
		lo, hi = min(es.min(), eo.min()), max(es.max(), eo.max())
		bins = np.linspace(lo - .5, hi + .5, 60)
		b.hist(eo, bins=bins, color=C_BAD, label=f'{len(eo)} no soluciones')
		b.hist(es, bins=bins, color=C_GOOD, label=f'{len(es)} soluciones')
		b.set_yscale('log'); b.set_xlabel('$\\langle |W|^2\\rangle$')
		sep = es.max() < eo.min()
		b.set_title(('SEPARA: soluciones = N exacto' if sep
					 else 'NO SEPARA: todo vale N'), fontsize=11,
					color=('#2b8a3e' if sep else '#c92a2a'))
		b.legend(fontsize=9)

	fig.suptitle(f'N={n}: detectar conflictos e identificar el tablero son incompatibles en 1D',
				 fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .94))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot(7))
