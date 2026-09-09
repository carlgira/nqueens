"""
Cuencas de atraccion: en vez de buscar la solucion, buscar la region desde la
que se cae en ella.

Potencial = nuestro testigo, conflictos = (<S^2> - N)/2.
Movimiento = intercambiar dos columnas (se mantiene la permutacion).
Descenso   = ir siempre al mejor vecino; parar en minimo local.
"""

import numpy as np
from itertools import permutations, combinations


def conflicts(p):
	n = len(p); i = np.arange(n)
	a = i + p; b = i - p
	return (n - len(set(a.tolist()))) + (n - len(set(b.tolist())))


def descend(p, max_steps=200):
	"""Descenso voraz por intercambios. Devuelve (final, pasos, atascado)."""
	p = np.asarray(p).copy(); n = len(p); c = conflicts(p)
	for step in range(max_steps):
		if c == 0:
			return tuple(p), step, False
		best, bi, bj = c, -1, -1
		for i, j in combinations(range(n), 2):
			p[i], p[j] = p[j], p[i]
			cc = conflicts(p)
			p[i], p[j] = p[j], p[i]
			if cc < best:
				best, bi, bj = cc, i, j
		if bi < 0:
			return tuple(p), step, True
		p[bi], p[bj] = p[bj], p[bi]; c = best
	return tuple(p), max_steps, True


def survey(n, sample=None, seed=0):
	rng = np.random.default_rng(seed)
	src = ([tuple(rng.permutation(n)) for _ in range(sample)] if sample
		   else list(permutations(range(n))))
	steps, ok, sinks = [], 0, {}
	for p in src:
		f, s, stuck = descend(p)
		if not stuck:
			ok += 1; steps.append(s); sinks[f] = sinks.get(f, 0) + 1
	return len(src), ok, np.array(steps), sinks


if __name__ == '__main__':
	print(f'{"N":>3} {"muestras":>9} {"llegan a solucion":>18} {"pasos: media":>13} '
		  f'{"max":>5} {"cuencas":>9}')
	print('-' * 64)
	for n, s in ((7, None), (8, None), (9, 20000), (10, 12000), (11, 8000), (12, 5000)):
		tot, ok, st, sinks = survey(n, s)
		print(f'{n:>3} {tot:>9,} {f"{ok:,} ({ok/tot:.0%})":>18} '
			  f'{st.mean():>13.2f} {st.max():>5} {len(sinks):>9,}')
