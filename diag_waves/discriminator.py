"""
¿Se puede aprender a distinguir un espectro REALIZABLE de uno fantasma?

Solucion vs permutacion cualquiera es trivial: el testigo lo resuelve exacto.
El problema abierto es dentro del conjunto factible por momentos -- todos esos
tienen c_k en {0,1}, misma energia, misma area, mismos momentos. Lo unico que
los separa es si existe una permutacion que los realice.

Red pequena en numpy puro, con particion entrenamiento/prueba honesta.
"""

import numpy as np

from diag_waves.bounds import feasible_pairs
from diag_waves.diag_waves import channels, shift


def sol_spectra(n):
	out, col, dp, dm, cur = set(), [False] * n, set(), set(), []
	def rec(r):
		if r == n:
			a, b = shift(*channels(list(cur)), n)
			out.add((tuple(sorted(a)), tuple(sorted(b)))); return
		for c in range(n):
			if col[c] or (r + c) in dp or (r - c) in dm: continue
			col[c] = True; dp.add(r + c); dm.add(r - c); cur.append(c)
			rec(r + 1); col[c] = False; dp.discard(r + c); dm.discard(r - c); cur.pop()
	rec(0)
	return out


def dataset(n):
	fp = feasible_pairs(n); real = sol_spectra(n)
	X = np.zeros((len(fp), 4 * n), dtype=np.float32)
	y = np.zeros(len(fp), dtype=np.float32)
	for k, (A, b) in enumerate(fp):
		for v in A: X[k, v] = 1
		for v in b: X[k, 2 * n + v] = 1
		y[k] = 1.0 if (A, b) in real else 0.0
	return X, y


def train(X, y, hidden=(96, 48), epochs=400, lr=3e-3, seed=0, frac=.75):
	rng = np.random.default_rng(seed)
	idx = rng.permutation(len(X)); cut = int(frac * len(X))
	tr, te = idx[:cut], idx[cut:]
	Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
	mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
	Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

	dims = [X.shape[1]] + list(hidden) + [1]
	W = [rng.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
		 for i in range(len(dims) - 1)]
	B = [np.zeros(d, dtype=np.float32) for d in dims[1:]]
	mW = [np.zeros_like(w) for w in W]; mB = [np.zeros_like(b) for b in B]

	def fwd(Z):
		acts = [Z]
		for i, (w, b) in enumerate(zip(W, B)):
			Z = Z @ w + b
			if i < len(W) - 1: Z = np.maximum(Z, 0)
			acts.append(Z)
		return acts

	for ep in range(epochs):
		p = rng.permutation(len(Xtr))
		for s in range(0, len(p), 256):
			bi = p[s:s + 256]
			acts = fwd(Xtr[bi])
			out = 1 / (1 + np.exp(-acts[-1][:, 0]))
			g = ((out - ytr[bi]) / len(bi))[:, None]
			for i in range(len(W) - 1, -1, -1):
				gw = acts[i].T @ g; gb = g.sum(0)
				if i > 0:
					g = (g @ W[i].T) * (acts[i] > 0)
				mW[i] = .9 * mW[i] + gw; mB[i] = .9 * mB[i] + gb
				W[i] -= lr * mW[i]; B[i] -= lr * mB[i]

	def acc(Z, t):
		o = 1 / (1 + np.exp(-fwd(Z)[-1][:, 0]))
		r = np.argsort(np.argsort(o)); n1 = t.sum(); n0 = len(t) - n1
		auc = (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
		return ((o > .5) == (t > .5)).mean(), auc

	return acc(Xtr, ytr), acc(Xte, yte), max(yte.mean(), 1 - yte.mean())


if __name__ == '__main__':
	for n in (10, 11, 12):
		X, y = dataset(n)
		(atr, utr), (ate, ute), base = train(X, y)
		print(f'N={n}: {len(X):,} espectros factibles, {int(y.sum()):,} realizables')
		print(f'   entrenamiento: acierto {atr:.1%}  AUC {utr:.3f}')
		print(f'   PRUEBA       : acierto {ate:.1%}  AUC {ute:.3f}   (mayoritaria {base:.1%})')
		print()
