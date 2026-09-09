"""
¿Hay clusters en los coeficientes de Fourier de las soluciones?

Los c_k son binarios (0/1), así que agrupar por ellos es agrupar patrones de
diagonales ocupadas. El riesgo es obvio: k-means SIEMPRE devuelve k clusters.
Todo aquí se compara contra un nulo que preserva las marginales <c_k> pero
destruye las correlaciones (barajado por coordenada).
"""

import numpy as np


def sol_coefs(n, nk=20):
	"""Los primeros nk coeficientes de cada espectro de solución (sin repetir)."""
	seen, cur, full = set(), [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if r == n:
			c = np.zeros(4 * n, dtype=np.int8)
			for i, s in enumerate(cur):
				c[i + s + 1] = 1; c[2 * n + i - s + n] = 1
			seen.add(c.tobytes()); return
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b; k = b.bit_length() - 1
			cur.append(k); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	M = np.array([np.frombuffer(s, dtype=np.int8) for s in seen], dtype=float)
	return M[:, 1:nk + 1]


def null_model(X, rng):
	"""Baraja cada columna por separado: conserva <c_k>, mata las correlaciones."""
	Y = X.copy()
	for j in range(Y.shape[1]):
		rng.shuffle(Y[:, j])
	return Y


def kmeans(X, k, rng, iters=60, restarts=8):
	best, best_lab, best_cen = np.inf, None, None
	for _ in range(restarts):
		cen = X[rng.choice(len(X), k, replace=False)].copy()
		for _ in range(iters):
			d = ((X[:, None, :] - cen[None]) ** 2).sum(-1)
			lab = d.argmin(1)
			for c in range(k):
				m = lab == c
				if m.any(): cen[c] = X[m].mean(0)
		inertia = ((X - cen[lab]) ** 2).sum()
		if inertia < best:
			best, best_lab, best_cen = inertia, lab, cen
	return best, best_lab


def silhouette(X, lab, rng, sample=600):
	idx = rng.choice(len(X), min(sample, len(X)), replace=False)
	D = np.sqrt(((X[idx][:, None] - X[idx][None]) ** 2).sum(-1))
	L = lab[idx]
	out = []
	for i in range(len(idx)):
		same = L == L[i]; same[i] = False
		if not same.any(): continue
		a = D[i, same].mean()
		b = min(D[i, L == c].mean() for c in set(L) - {L[i]})
		out.append((b - a) / max(a, b))
	return float(np.mean(out))


def report(n=12, nk=20, seed=0):
	rng = np.random.default_rng(seed)
	X = sol_coefs(n, nk)
	Y = null_model(X, rng)
	print(f'N={n}: {len(X)} espectros de solución, {nk} coeficientes\n')

	# distribución de distancias
	s = rng.choice(len(X), min(700, len(X)), replace=False)
	for name, M in (('soluciones', X), ('nulo', Y)):
		D = np.sqrt(((M[s][:, None] - M[s][None]) ** 2).sum(-1))
		d = D[np.triu_indices(len(s), 1)]
		print(f'  distancias {name:>11}: media {d.mean():.3f}  sd {d.std():.3f}  '
			  f'rango [{d.min():.2f}, {d.max():.2f}]')

	print(f'\n  {"k":>3} {"silueta soluciones":>20} {"silueta nulo":>14} {"veredicto":>12}')
	for k in (2, 3, 4, 5, 6, 8):
		_, lx = kmeans(X, k, rng); _, ly = kmeans(Y, k, rng)
		sx, sy = silhouette(X, lx, rng), silhouette(Y, ly, rng)
		v = 'estructura' if sx > sy + 0.05 else 'nada'
		print(f'  {k:>3} {sx:>20.4f} {sy:>14.4f} {v:>12}')


if __name__ == '__main__':
	report(12)
