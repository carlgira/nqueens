"""
Predecir si una rama del backtracking esta CONDENADA, y podarla antes de bajar.

Se enumera el arbol completo de N-reinas, se etiqueta cada estado parcial
(profundidad N//2 .. N-1) con si su subarbol contiene alguna solucion, y se
entrena una red. Luego se mide la poda REAL: nodos ahorrados vs soluciones perdidas.
"""

import numpy as np


def harvest(n, dmin=None, dmax=None):
	"""Todos los estados parciales del arbol, etiquetados por si llevan a solucion."""
	dmin = dmin or n // 2; dmax = dmax or n - 1
	full = (1 << n) - 1
	rows = []
	def rec(r, cols, dp, dm, cur):
		if r == n: return 1
		tot = 0
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b; c = b.bit_length() - 1
			cur.append(c)
			tot += rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1, cur)
			cur.pop()
		if dmin <= r <= dmax:
			rows.append((tuple(cur), tot > 0))
		return tot
	rec(0, 0, 0, 0, [])
	return rows


def features(cur, n):
	"""Rasgos del estado parcial: lo colocado (espectro) y lo que queda libre."""
	d = len(cur)
	cols = set(cur); dp = {i + cur[i] for i in range(d)}; dm = {i - cur[i] for i in range(d)}
	rowdeg = []; free = []
	for r in range(d, n):
		opts = [c for c in range(n) if c not in cols and (r + c) not in dp and (r - c) not in dm]
		rowdeg.append(len(opts)); free += [(r, c) for c in opts]
	nf = len(free)
	hp = np.zeros(2 * n); hm = np.zeros(2 * n); hc = np.zeros(n)
	for i, c in free:
		hp[(i + c) % (2 * n)] += 1; hm[(i - c) % (2 * n)] += 1; hc[c] += 1
	rd = np.zeros(n); rd[:len(rowdeg)] = rowdeg
	sp = np.zeros(2 * n); sm = np.zeros(2 * n)
	for i in range(d):
		sp[i + cur[i]] += 1; sm[i - cur[i] + n - 1] += 1
	e = 1e-9
	return np.concatenate([
		[d / n, nf / (n * n), min(rowdeg or [0]) / n, np.mean(rowdeg or [0]) / n],
		[(hp ** 2).sum() / (nf ** 2 + e), (hm ** 2).sum() / (nf ** 2 + e),
		 (hc ** 2).sum() / (nf ** 2 + e)],
		hp / (nf + e), hm / (nf + e), hc / (nf + e), rd / n, sp, sm,
	]).astype(np.float32)


def simple_features(cur, n):
	"""7 rasgos ADIMENSIONALES: valen igual para cualquier N."""
	d = len(cur); cols = set(cur)
	dp = {i + cur[i] for i in range(d)}; dm = {i - cur[i] for i in range(d)}
	rowdeg = []; hp = {}; hm = {}; hc = {}; nf = 0
	for r in range(d, n):
		k = 0
		for c in range(n):
			if c in cols or (r + c) in dp or (r - c) in dm: continue
			k += 1; nf += 1
			hp[r + c] = hp.get(r + c, 0) + 1
			hm[r - c] = hm.get(r - c, 0) + 1
			hc[c] = hc.get(c, 0) + 1
		rowdeg.append(k)
	e = 1e-9
	return np.array([d / n, nf / (n * n), min(rowdeg or [0]) / n,
					 np.mean(rowdeg or [0]) / n,
					 sum(v * v for v in hp.values()) / (nf ** 2 + e),
					 sum(v * v for v in hm.values()) / (nf ** 2 + e),
					 sum(v * v for v in hc.values()) / (nf ** 2 + e)], dtype=np.float32)


def sample_states(n, per_depth=3000, dmin=None, dmax=None, seed=0):
	"""Estados parciales por caminos aleatorios, etiquetados por si son extensibles."""
	rng = np.random.default_rng(seed)
	dmin = dmin or n // 2; dmax = dmax or n - 1
	full = (1 << n) - 1
	def extendable(r, cols, dp, dm):
		if r == n: return True
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b
			if extendable(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1): return True
		return False
	out = []
	for d in range(dmin, dmax + 1):
		got = 0; guard = 0
		while got < per_depth and guard < per_depth * 40:
			guard += 1
			cols = dp = dm = 0; cur = []
			ok = True
			for r in range(d):
				free = ~(cols | dp | dm) & full
				if not free: ok = False; break
				bits = []
				while free:
					b = free & -free; free ^= b; bits.append(b)
				b = bits[rng.integers(len(bits))]
				cur.append(b.bit_length() - 1)
				cols |= b; dp = (dp | b) << 1; dm = (dm | b) >> 1
			if not ok: continue
			out.append((tuple(cur), extendable(d, cols, dp, dm))); got += 1
	return out
