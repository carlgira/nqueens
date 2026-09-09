"""
La condicion de Hall COMPLETA, via emparejamiento bipartito.

Hall: existe emparejamiento perfecto <=> todo subconjunto tiene suficientes
vecinos. Comprobarlo con un emparejamiento equivale a comprobar los 2^(N-r)
subconjuntos, en O(V*E).

Ojo: hay TRES restricciones (columnas, d+, d-) y satisfacerlas a la vez es
emparejamiento 3-dimensional, que es NP-duro. Asi que se comprueba cada una
por separado: cada una es exacta y necesaria, las tres juntas no bastan.
"""

import numpy as np


def kuhn(adj, n_right):
	"""¿hay emparejamiento perfecto que cubra todas las filas de adj?"""
	match = [-1] * n_right
	def try_row(u, seen):
		for v in adj[u]:
			if seen[v]: continue
			seen[v] = True
			if match[v] == -1 or try_row(match[v], seen):
				match[v] = u; return True
		return False
	for u in range(len(adj)):
		if not try_row(u, [False] * n_right):
			return False
	return True


def available(cur, n):
	d = len(cur); cols = set(cur)
	dp = {i + cur[i] for i in range(d)}; dm = {i - cur[i] for i in range(d)}
	out = []
	for r in range(d, n):
		out.append([(c, r + c, r - c + n) for c in range(n)
					if c not in cols and (r + c) not in dp and (r - c) not in dm])
	return out


def hall_full(cur, n):
	"""True si esta CONDENADO segun alguna de las tres condiciones de Hall."""
	av = available(cur, n)
	if any(not row for row in av): return True
	for idx, size in ((0, n), (1, 2 * n), (2, 2 * n + 1)):
		adj = [[t[idx] for t in row] for row in av]
		if not kuhn(adj, size): return True
	return False
