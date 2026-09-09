"""La forma de las cuencas: no son bolas, son objetos porosos que se desvanecen."""

import numpy as np
from itertools import permutations, combinations

from diag_waves.fast_basins import Board


def cayley(p, q):
	n = len(p); pos = {v: i for i, v in enumerate(q)}
	perm = [pos[v] for v in p]
	seen = [False] * n; c = 0
	for i in range(n):
		if not seen[i]:
			c += 1; j = i
			while not seen[j]: seen[j] = True; j = perm[j]
	return n - c


def basins(n=8):
	pairs = list(combinations(range(n), 2))
	def desc(p):
		b = Board(np.array(p))
		for _ in range(50):
			if b.c == 0: return tuple(int(v) for v in b.p)
			best, mv = b.c + 1, None
			for i, j in pairs:
				d = b.delta(i, j)
				if d < best: best, mv = d, (i, j)
			if mv is None or best >= b.c: return None
			b.swap(*mv)
		return None
	out = {}
	for p in permutations(range(n)):
		f = desc(p)
		if f is not None: out.setdefault(f, set()).add(p)
	return out, pairs


def plot(n=8, path='diag_waves/fig_shape.png'):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	B, pairs = basins(n)
	perms = list(permutations(range(n)))
	shells = np.array([len([1 for p in perms if cayley(p, perms[0]) == d]) for d in range(n)])

	fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
	C1, C2 = '#e8590c', '#495057'
	keys = sorted(B, key=lambda k: -len(B[k]))[:1] + [sorted(B, key=lambda k: len(B[k]))[0]]

	for sol, col, lab in zip(keys, (C1, '#1c7ed6'), ('cuenca grande', 'cuenca pequeña')):
		S = B[sol]
		fill = np.zeros(n)
		for d in range(n):
			tot = shells[d]
			if tot: fill[d] = sum(1 for p in S if cayley(p, sol) == d) / tot
		ax[0].plot(range(n), fill, 'o-', color=col, lw=2, label=f'{lab} ({len(S)})')
		cum = np.cumsum([sum(1 for p in S if cayley(p, sol) == d) for d in range(n)])
		ax[1].plot(range(n), cum, 'o-', color=col, lw=2, label=lab)
		bal = np.cumsum(shells)
		k = np.searchsorted(bal, len(S))
		ax[1].plot(range(n), np.minimum(bal, len(S)), '--', color=col, lw=1.2, alpha=.6,
				   label=f'bola del mismo tamaño')
	ax[0].axhline(1, color='k', ls=':', lw=1)
	ax[0].text(3.2, 1.02, 'una BOLA sería 100% hasta su radio', fontsize=9)
	ax[0].set_xlabel('distancia de Cayley (nº de intercambios)')
	ax[0].set_ylabel('fracción de la capa que está en la cuenca')
	ax[0].set_title('El llenado se desvanece:\nninguna capa está completa', fontsize=11)
	ax[0].set_xlim(0, 6); ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)
	ax[1].set_xlabel('distancia de Cayley'); ax[1].set_ylabel('acumulado')
	ax[1].set_xlim(0, 6); ax[1].set_title('Contra una bola del mismo tamaño', fontsize=11)
	ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)

	por = []
	for sol in list(B)[:20]:
		S = B[sol]
		for p in S:
			k = 0
			for i, j in pairs:
				q = list(p); q[i], q[j] = q[j], q[i]
				k += tuple(q) in S
			por.append(k / len(pairs))
	ax[2].hist(por, bins=28, color=C1)
	ax[2].axvline(np.mean(por), color='k', ls='--', lw=1.5,
				  label=f'media {np.mean(por):.0%}')
	ax[2].set_xlabel('fracción de vecinos que están en la MISMA cuenca')
	ax[2].set_ylabel('puntos')
	ax[2].set_title(f'Porosidad: cada punto tiene {len(pairs)} vecinos\n'
					'casi todos apuntan fuera', fontsize=11)
	ax[2].legend(fontsize=9)

	fig.suptitle(f'N={n}: la forma de las cuencas — no son esferas', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	print(f'porosidad media: {np.mean(por):.1%} de los vecinos siguen en la cuenca')
	return path


if __name__ == '__main__':
	print(plot())
