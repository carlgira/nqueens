"""
Autoencoder sobre las soluciones: comprimir a un espacio latente, muestrear
ruido gaussiano ahi, y decodificar para GENERAR tableros.

Representacion: la matriz de permutacion NxN aplanada.
Decodificacion: asignacion voraz (la casilla mas caliente disponible).
Todo en numpy, retropropagacion a mano.
"""

import numpy as np


def solutions(n, cap=None):
	out, cur, full = [], [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if cap and len(out) >= cap: return
		if r == n: out.append(tuple(cur)); return
		f = ~(cols | dp | dm) & full
		while f:
			b = f & -f; f ^= b; c = b.bit_length() - 1
			cur.append(c); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	return out


def to_matrix(perms, n):
	X = np.zeros((len(perms), n * n), dtype=np.float32)
	for k, p in enumerate(perms):
		X[k, np.arange(n) * n + np.asarray(p)] = 1
	return X


def greedy_perm(soft, n):
	"""matriz blanda NxN -> permutacion, cogiendo siempre la casilla mas alta."""
	M = soft.reshape(n, n).copy()
	rows = set(range(n)); cols = set(range(n)); p = [-1] * n
	for _ in range(n):
		best = None
		for r in rows:
			for c in cols:
				if best is None or M[r, c] > M[best[0], best[1]]: best = (r, c)
		p[best[0]] = best[1]; rows.discard(best[0]); cols.discard(best[1])
	return tuple(p)


def is_sol(p):
	n = len(p); i = np.arange(n)
	return len(set((i + p).tolist())) == n and len(set((i - p).tolist())) == n


class AE:
	def __init__(self, d_in, hid, lat, rng):
		dims = [d_in, hid, lat, hid, d_in]
		self.W = [rng.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
				  for i in range(4)]
		self.B = [np.zeros(dims[i + 1], dtype=np.float32) for i in range(4)]
		self.mW = [np.zeros_like(w) for w in self.W]
		self.mB = [np.zeros_like(b) for b in self.B]
		self.lat = lat

	def fwd(self, Z):
		a = [Z]
		for i in range(4):
			Z = Z @ self.W[i] + self.B[i]
			if i < 3: Z = np.maximum(Z, 0)
			a.append(Z)
		return a

	def encode(self, Z):
		return self.fwd(Z)[3 - 1]			# salida de la capa latente (indice 2)

	def decode(self, L):
		Z = np.maximum(L @ self.W[2] + self.B[2], 0)
		return 1 / (1 + np.exp(-(Z @ self.W[3] + self.B[3])))

	def fit(self, X, epochs=300, lr=2e-3, bs=256, rng=None):
		for _ in range(epochs):
			for s in range(0, len(X), bs):
				bi = rng.integers(0, len(X), bs)
				a = self.fwd(X[bi])
				out = 1 / (1 + np.exp(-a[4]))
				g = (out - X[bi]) / bs
				for i in range(3, -1, -1):
					gw = a[i].T @ g; gb = g.sum(0)
					if i > 0:
						g = g @ self.W[i].T
						if i > 1: g = g * (a[i] > 0)
					self.mW[i] = .9 * self.mW[i] + gw; self.mB[i] = .9 * self.mB[i] + gb
					self.W[i] -= lr * self.mW[i]; self.B[i] -= lr * self.mB[i]


def sigmoid(z):
	return np.where(z >= 0, 1 / (1 + np.exp(-np.clip(z, -60, 60))),
					np.exp(np.clip(z, -60, 60)) / (1 + np.exp(np.clip(z, -60, 60))))


class VAE:
	"""Fuerza que el latente sea N(0,I), asi muestrear ruido SI cae en la variedad."""

	def __init__(self, d_in, hid, lat, rng):
		def w(a, b): return rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
		self.We, self.be = w(d_in, hid), np.zeros(hid, np.float32)
		self.Wm, self.bm = w(hid, lat), np.zeros(lat, np.float32)
		self.Wv, self.bv = w(hid, lat), np.zeros(lat, np.float32)
		self.Wd, self.bd = w(lat, hid), np.zeros(hid, np.float32)
		self.Wo, self.bo = w(hid, d_in), np.zeros(d_in, np.float32)
		self.lat = lat
		self.par = ['We', 'be', 'Wm', 'bm', 'Wv', 'bv', 'Wd', 'bd', 'Wo', 'bo']
		self.mom = {k: np.zeros_like(getattr(self, k)) for k in self.par}

	def decode(self, z):
		h = np.maximum(z @ self.Wd + self.bd, 0)
		return sigmoid(h @ self.Wo + self.bo)

	def fit(self, X, epochs=400, lr=2e-3, bs=256, beta=1.0, rng=None):
		for ep in range(epochs):
			for _ in range(max(1, len(X) // bs)):
				bi = rng.integers(0, len(X), bs)
				x = X[bi]
				h = np.maximum(x @ self.We + self.be, 0)
				mu = h @ self.Wm + self.bm
				lv = np.clip(h @ self.Wv + self.bv, -8, 8)
				eps = rng.standard_normal(mu.shape).astype(np.float32)
				z = mu + eps * np.exp(.5 * lv)
				hd = np.maximum(z @ self.Wd + self.bd, 0)
				out = sigmoid(hd @ self.Wo + self.bo)

				go = (out - x) / bs
				gWo = hd.T @ go; gbo = go.sum(0)
				ghd = (go @ self.Wo.T) * (hd > 0)
				gWd = z.T @ ghd; gbd = ghd.sum(0)
				gz = ghd @ self.Wd.T
				gmu = gz + beta * mu / bs
				glv = gz * eps * .5 * np.exp(.5 * lv) + beta * .5 * (np.exp(lv) - 1) / bs
				gWm = h.T @ gmu; gbm = gmu.sum(0)
				gWv = h.T @ glv; gbv = glv.sum(0)
				gh = ((gmu @ self.Wm.T) + (glv @ self.Wv.T)) * (h > 0)
				gWe = x.T @ gh; gbe = gh.sum(0)

				for k, g in zip(self.par, [gWe, gbe, gWm, gbm, gWv, gbv, gWd, gbd, gWo, gbo]):
					self.mom[k] = .9 * self.mom[k] + g
					setattr(self, k, getattr(self, k) - lr * self.mom[k])
