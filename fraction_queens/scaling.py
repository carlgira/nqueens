"""
Como escala la separacion con N.

Para cada N: C = ln(max a_i)/ln q de
  - la construccion clasica de wave_queens (una por clase de N mod 6) y su
    transpuesta,
  - soluciones genericas (backtracking con orden aleatorio),
  - permutaciones al azar,
  - progresiones aritmeticas (sean o no solucion).
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import frac
from cf import measures
from law import arithmetic
from wave_queens.constructions import modular, even, mod6_3

NS = list(range(8, 61))


def classic(n):
	if n % 6 == 3:
		return mod6_3(n)
	if n % 2 == 0:
		return even(n)
	return modular(n)


def inverse(s):
	o = [0] * len(s)
	for k, v in enumerate(s):
		o[v] = k
	return tuple(o)


def random_solution(n, rng):
	col, d1, d2 = set(), set(), set()
	cur = [0] * n

	def go(k):
		if k == n:
			return True
		order = list(range(n))
		rng.shuffle(order)
		for c in order:
			if c in col or (k + c) in d1 or (k - c) in d2:
				continue
			col.add(c); d1.add(k + c); d2.add(k - c); cur[k] = c
			if go(k + 1):
				return True
			col.discard(c); d1.discard(k + c); d2.discard(k - c)
		return False
	go(0)
	return tuple(cur)


def run():
	rng = random.Random(7)
	rows = []
	for n in NS:
		c = classic(n)
		if c is None:
			continue
		C = lambda s: measures(s, n)[1]
		best_ar = max(C(arithmetic(n, a, k)) for k in range(1, n)
		              if math.gcd(k, n) == 1 for a in range(n))
		gen = np.mean([C(random_solution(n, rng)) for _ in range(8)])
		per = np.mean([C(tuple(rng.sample(range(n), n))) for _ in range(8)])
		rows.append((n, C(c), C(inverse(c)), best_ar, gen, per))
	return np.array(rows)


if __name__ == '__main__':
	R = run()
	np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaling.npy'), R)
	print(f"{'N':>4} {'clasica':>8} {'transp':>8} {'mejor arit':>11} "
	      f"{'sol generica':>13} {'perm azar':>10}")
	for r in R:
		print(f"{int(r[0]):>4} {r[1]:>8.3f} {r[2]:>8.3f} {r[3]:>11.3f} "
		      f"{r[4]:>13.3f} {r[5]:>10.3f}")
