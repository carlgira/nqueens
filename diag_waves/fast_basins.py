"""Descenso rapido: los conflictos se actualizan en O(1) por movimiento.

Al quitar una reina de una diagonal con k ocupantes, los conflictos bajan k-1.
Al ponerla en una con k, suben k. Eso hace cada swap O(1) y cada paso O(N^2).
"""

import numpy as np


class Board:
	def __init__(self, p):
		self.n = n = len(p)
		self.p = np.asarray(p).copy()
		self.dp = np.zeros(2 * n - 1, dtype=np.int64)
		self.dm = np.zeros(2 * n - 1, dtype=np.int64)
		for r in range(n):
			self.dp[r + self.p[r]] += 1
			self.dm[r - self.p[r] + n - 1] += 1
		self.c = int((self.dp * (self.dp - 1) // 2).sum() +
					 (self.dm * (self.dm - 1) // 2).sum())

	def _rem(self, r, c):
		a = r + c; b = r - c + self.n - 1
		self.dp[a] -= 1; self.c -= self.dp[a]
		self.dm[b] -= 1; self.c -= self.dm[b]

	def _add(self, r, c):
		a = r + c; b = r - c + self.n - 1
		self.c += self.dp[a]; self.dp[a] += 1
		self.c += self.dm[b]; self.dm[b] += 1

	def swap(self, i, j):
		ci, cj = self.p[i], self.p[j]
		self._rem(i, ci); self._rem(j, cj)
		self.p[i], self.p[j] = cj, ci
		self._add(i, cj); self._add(j, ci)

	def delta(self, i, j):
		self.swap(i, j); d = self.c; self.swap(i, j)
		return d


def descend(p, rng, side=0, cap=None):
	"""Devuelve (solucion|None, pasos). side=0 -> descenso puro."""
	b = Board(p); n = b.n; cap = cap or 30 * n; used = 0
	for step in range(cap):
		if b.c == 0:
			return tuple(int(v) for v in b.p), step
		best, moves = b.c + 1, []
		for i in range(n - 1):
			for j in range(i + 1, n):
				d = b.delta(i, j)
				if d < best: best, moves = d, [(i, j)]
				elif d == best: moves.append((i, j))
		if best > b.c: return None, step
		if best == b.c:
			used += 1
			if used > side: return None, step
		i, j = moves[rng.integers(len(moves))]
		b.swap(i, j)
	return None, cap
