"""
El embudo geometrico: de N! permutaciones a las Q(N) soluciones, aplicando
solo condiciones sobre la CURVA. Todos los umbrales se calculan con N solo,
enumerando los espectros factibles -- nunca se miran las soluciones.
"""

import numpy as np
from itertools import permutations, islice

from diag_waves.diag_waves import band_series
from diag_waves.bounds import feasible_pairs, pair_wave


def thresholds(n, G=200):
	x = np.linspace(0, 2 * np.pi, G)
	body = (x > .5) & (x < 2 * np.pi - .5)
	fw = np.array([pair_wave(p, n, x) for p in feasible_pairs(n)])
	return x, body, fw[:, body].max(), fw[:, body].min(), (fw ** 4).mean(1).max()


def funnel(n, sample=None, chunk=40000, seed=0, G=200):
	x, body, techo, piso, s4max = thresholds(n, G)
	B = np.cos(np.outer(np.arange(4 * n), x)).astype(np.float32)
	counts = np.zeros(4, dtype=np.int64); tot = 0
	rng = np.random.default_rng(seed)

	def batch(C):
		nonlocal tot
		tot += len(C)
		S = C @ B
		k = S[:, body].max(1) <= techo + 1e-6
		counts[0] += k.sum()
		k &= S[:, body].min(1) >= piso - 1e-6
		counts[1] += k.sum()
		k &= (S ** 4).mean(1) <= s4max + 1e-4
		counts[2] += k.sum()
		k &= np.isclose(.5 * (C ** 2).sum(1), n)
		counts[3] += k.sum()

	if sample:
		for _ in range(0, sample, chunk):
			m = min(chunk, sample - tot)
			batch(np.array([band_series(rng.permutation(n)) for _ in range(m)], np.float32))
	else:
		it = permutations(range(n))
		while True:
			blk = list(islice(it, chunk))
			if not blk: break
			batch(np.array([band_series(p) for p in blk], np.float32))
	return tot, counts, (techo, piso, s4max)


if __name__ == '__main__':
	from math import factorial
	Q = {7:40, 8:92, 9:352, 10:724, 11:2680, 12:14200}
	print(f'{"N":>3} {"muestras":>12} {"+ techo":>18} {"+ piso":>18} '
		  f'{"+ <S^4>":>18} {"+ testigo":>16}')
	print('-' * 92)
	for n, s in ((7, None), (8, None), (9, None), (10, None), (11, 3_000_000), (12, 3_000_000)):
		tot, c, th = funnel(n, sample=s)
		f = [f'{v:>9,} {v/tot:>7.3%}' for v in c]
		tag = '' if s is None else '  (muestra)'
		print(f'{n:>3} {tot:>12,} ' + ' '.join(f'{v:>17}' for v in f) + tag)
		if s is None:
			assert c[3] == Q[n], (c[3], Q[n])
