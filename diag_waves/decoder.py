"""
Hélice -> permutación.  Entrada: media hélice. Salida: el tablero.

OJO: distintas soluciones comparten hélice (el espectro pierde el emparejamiento),
asi que hay un TECHO estructural: un decodificador determinista solo puede acertar
la mas frecuente de cada grupo.
"""

import numpy as np
from collections import defaultdict

from diag_waves.diag_waves import band_series, channels, conflicts


def solutions(n):
	out, cur, full = [], [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if r == n: out.append(tuple(cur)); return
		f = ~(cols | dp | dm) & full
		while f:
			b = f & -f; f ^= b; c = b.bit_length() - 1
			cur.append(c); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	return out


def half_helix(perms, n):
	"""Media hélice: Re y Im en x de 0 a pi. Contiene toda la información."""
	M = 4 * n
	x = 2 * np.pi * np.arange(M // 2 + 1) / M
	E = np.exp(1j * np.outer(np.arange(M), x))
	C = np.array([band_series(p) for p in perms])
	Z = C @ E
	return np.hstack([Z.real, Z.imag]).astype(np.float32)


def ceiling(perms, n):
	"""Mejor acierto posible de un decodificador determinista."""
	g = defaultdict(list)
	for p in perms:
		g[band_series(p).tobytes()].append(p)
	best = sum(max(1, len(v)) and 1 for v in g.values())
	return len(g) / len(perms), np.mean([len(v) for v in g.values()])


class Decoder:
	def __init__(self, d_in, hid, n, rng):
		def w(a, b): return rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
		self.n = n
		self.W1, self.b1 = w(d_in, hid), np.zeros(hid, np.float32)
		self.W2, self.b2 = w(hid, hid), np.zeros(hid, np.float32)
		self.W3, self.b3 = w(hid, n * n), np.zeros(n * n, np.float32)
		self.par = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
		self.mom = {k: np.zeros_like(getattr(self, k)) for k in self.par}

	def fwd(self, X):
		h1 = np.maximum(X @ self.W1 + self.b1, 0)
		h2 = np.maximum(h1 @ self.W2 + self.b2, 0)
		o = (h2 @ self.W3 + self.b3).reshape(-1, self.n, self.n)
		o = o - o.max(2, keepdims=True)
		p = np.exp(o); p /= p.sum(2, keepdims=True)		# softmax por fila
		return h1, h2, p

	def fit(self, X, Y, epochs=300, lr=2e-3, bs=256, rng=None):
		n = self.n
		for _ in range(epochs):
			for _ in range(max(1, len(X) // bs)):
				bi = rng.integers(0, len(X), bs)
				x = X[bi]; h1, h2, p = self.fwd(x)
				g = (p - Y[bi]) / bs
				g3 = g.reshape(bs, -1)
				gW3 = h2.T @ g3; gb3 = g3.sum(0)
				gh2 = (g3 @ self.W3.T) * (h2 > 0)
				gW2 = h1.T @ gh2; gb2 = gh2.sum(0)
				gh1 = (gh2 @ self.W2.T) * (h1 > 0)
				gW1 = x.T @ gh1; gb1 = gh1.sum(0)
				for k, gr in (('W3', gW3), ('b3', gb3), ('W2', gW2), ('b2', gb2),
							  ('W1', gW1), ('b1', gb1)):
					self.mom[k] = .9 * self.mom[k] + gr
					setattr(self, k, getattr(self, k) - lr * self.mom[k])


def decode(p, n):
	"""softmax NxN -> permutacion, asignacion voraz."""
	M = p.copy(); rows = set(range(n)); cols = set(range(n)); out = [-1] * n
	for _ in range(n):
		best = None
		for r in rows:
			for c in cols:
				if best is None or M[r, c] > M[best[0], best[1]]: best = (r, c)
		out[best[0]] = best[1]; rows.discard(best[0]); cols.discard(best[1])
	return tuple(out)
