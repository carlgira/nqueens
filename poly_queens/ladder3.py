"""
La escalera de niveles con los cumulos combinatorios: nivel 3 hasta N=27.

conflict_poly.py solo llegaba a N=12 porque necesitaba las N! permutaciones.
cumulants.py da kappa_[1..3] exactos sin enumerarlas, asi que la captura del
nivel 3 se puede medir en todo el rango donde Q(N) es conocido (A000170, N<=27).

	log Q(N)/N!  =  sum_k kappa_[k] (-1)^k / k!
	nivel 2  =  primer termino          =  -kappa_[1]
	nivel 3  =  hasta k=3               =  -kappa_[1] + kappa_[2]/2 - kappa_[3]/6
"""

import json
import os
from fractions import Fraction
from math import lgamma, log

QN = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596,
      2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884,
      314666222712, 2691008701644, 24233937684440, 227514171973736,
      2207893435808352, 22317699616364044, 234907967154122528]

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "kappa.json")


def table(nmax=31):
	"""kappa_[1..3] exactos, cacheados en kappa.json."""
	data = {}
	if os.path.exists(CACHE):
		data = {int(k): [Fraction(x) for x in v] for k, v in json.load(open(CACHE)).items()}
	from .cumulants import cumulants
	for n in range(6, nmax + 1):
		if n not in data:
			L = cumulants(n)
			data[n] = [L[1], L[2] * 2, L[3] * 6]
			print(f"  N={n} calculado", flush=True)
	json.dump({str(k): [str(x) for x in v] for k, v in data.items()},
	          open(CACHE, "w"))
	return data


def report(data):
	print(" N     t1       t2       t3    | nivel 2  nivel 3 | t3/N     Richardson")
	prev = {}
	for n in sorted(data):
		k1, k2, k3 = data[n]
		t = [-float(k1), float(k2) / 2, -float(k3) / 6]
		y = t[2] / n
		rich = ""
		if n - 2 in prev:
			rich = f"{(n*y - (n-2)*prev[n-2])/2:10.5f}"
		prev[n] = y
		if n <= len(QN):
			true = log(QN[n - 1]) - lgamma(n + 1)
			c2 = 100 * t[0] / true
			c3 = 100 * (t[0] + t[1] + t[2]) / true
			cap = f"{c2:7.1f}% {c3:7.1f}%"
		else:
			cap = "   Q(N) desconocido"
		print(f"{n:2d} {t[0]:8.3f} {t[1]:8.3f} {t[2]:8.3f} | {cap} | {y:8.5f} {rich}")


if __name__ == "__main__":
	import sys
	report(table(int(sys.argv[1]) if len(sys.argv) > 1 else 31))
