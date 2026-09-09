"""
Red multitarea: entrada = |Z| (la helice sin fase), dos salidas.

  cabeza A: reconstruir c_k          <- la forma estructurada (recuperacion de fase)
  cabeza B: realizable si/no          <- la etiqueta

La tarea A regulariza a la B: obliga al tronco a codificar QUE diagonales hay,
no un atajo cualquiera. Y de paso mide si |Z| determina c (problema del turnpike).
"""

import numpy as np


def sigmoid(z):
	z = np.clip(z, -60, 60)
	return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class MultiTask:
	def __init__(self, d_in, trunk, d_struct, rng):
		def w(a, b): return rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
		h1, h2 = trunk
		self.p = dict(
			W1=w(d_in, h1), b1=np.zeros(h1, np.float32),
			W2=w(h1, h2), b2=np.zeros(h2, np.float32),
			Wa=w(h2, d_struct), ba=np.zeros(d_struct, np.float32),
			Wb=w(h2, 1), bb=np.zeros(1, np.float32))
		self.m = {k: np.zeros_like(v) for k, v in self.p.items()}

	def n_params(self):
		return sum(v.size for v in self.p.values())

	def fwd(self, X):
		p = self.p
		h1 = np.maximum(X @ p['W1'] + p['b1'], 0)
		h2 = np.maximum(h1 @ p['W2'] + p['b2'], 0)
		return h1, h2, sigmoid(h2 @ p['Wa'] + p['ba']), sigmoid(h2 @ p['Wb'] + p['bb'])[:, 0]

	def fit(self, X, S, y, epochs=300, lr=2e-3, bs=256, wa=1.0, rng=None):
		p, m = self.p, self.m
		for _ in range(epochs):
			for _ in range(max(1, len(X) // bs)):
				bi = rng.integers(0, len(X), bs)
				x, s, t = X[bi], S[bi], y[bi]
				h1, h2, oa, ob = self.fwd(x)
				ga = wa * (oa - s) / bs
				gb = ((ob - t) / bs)[:, None]
				gWa = h2.T @ ga; gba = ga.sum(0)
				gWb = h2.T @ gb; gbb = gb.sum(0)
				gh2 = (ga @ p['Wa'].T + gb @ p['Wb'].T) * (h2 > 0)
				gW2 = h1.T @ gh2; gb2 = gh2.sum(0)
				gh1 = (gh2 @ p['W2'].T) * (h1 > 0)
				gW1 = x.T @ gh1; gb1 = gh1.sum(0)
				for k, g in (('Wa', gWa), ('ba', gba), ('Wb', gWb), ('bb', gbb),
							 ('W2', gW2), ('b2', gb2), ('W1', gW1), ('b1', gb1)):
					m[k] = .9 * m[k] + g; p[k] -= lr * m[k]


def auc(s, y):
	r = np.argsort(np.argsort(s)); n1 = y.sum(); n0 = len(y) - n1
	return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
