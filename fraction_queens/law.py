"""
De donde sale la cola: las progresiones aritmeticas.

(1) Que permutaciones sigma(k)=(a+ck) mod N caen a menos de N^-N de un
    racional de denominador D=(N+1)(N-1)^2  (medido, no supuesto).
(2) Toda progresion aritmetica -- sea solucion o no -- tiene C alto.
(3) Cuantas hay entre las soluciones y cuantas entre las permutaciones.
"""
import sys, os, math
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import frac
from cf import measures


def arithmetic(n, a, c):
	return tuple((a + c * k) % n for k in range(n))


def close_to(sigma, n, D):
	p, q = frac.reduced(sigma, n)
	x = Fraction(p, q)
	m = round(x * D)
	return abs(x - Fraction(m, D)) < Fraction(1, n ** n), m


if __name__ == '__main__':
	print("(1) que (a,c) caen a menos de N^-N de m/((N+1)(N-1)^2)")
	for n in (11, 12, 13, 20, 21, 25):
		D = (n + 1) * (n - 1) ** 2
		good = [(a, c) for c in range(1, n) if math.gcd(c, n) == 1
		        for a in range(n) if close_to(arithmetic(n, a, c), n, D)[0]]
		cs = sorted({c for _, c in good})
		print(f"  N={n:>3} D={D:>7}  pasos c={cs}  "
		      f"(= (N-1)/2, (N+1)/2, N-1)   {len(good)} pares (a,c)")

	print("\n(2) C = ln(max a_i)/ln q  de las progresiones aritmeticas vs al azar")
	rng = np.random.default_rng(0)
	print(f"{'N':>4} {'C_arit medio':>13} {'C_arit max':>11} {'C_azar medio':>13} "
	      f"{'C_azar max':>11}")
	for n in range(8, 31):
		ar = [arithmetic(n, a, c) for c in range(1, n) if math.gcd(c, n) == 1
		      for a in range(n)]
		rp = [tuple(int(v) for v in rng.permutation(n)) for _ in range(len(ar))]
		A = np.array([measures(s, n)[1] for s in ar])
		R = np.array([measures(s, n)[1] for s in rp])
		print(f"{n:>4} {A.mean():>13.3f} {A.max():>11.3f} {R.mean():>13.3f} "
		      f"{R.max():>11.3f}")

	print("\n(3) enriquecimiento: P(aritmetica | solucion) / P(aritmetica | permutacion)")
	print(f"{'N':>4} {'Q(N)':>8} {'arit.&sol':>10} {'%en sol':>9} {'%en perm':>11} {'x veces':>10}")
	for n in range(5, 14):
		sols = set(frac.solutions(n))
		ar = {arithmetic(n, a, c) for c in range(1, n) if math.gcd(c, n) == 1
		      for a in range(n)}
		hit = len(sols & ar)
		p_sol, p_per = hit / len(sols), len(ar) / math.factorial(n)
		print(f"{n:>4} {len(sols):>8} {hit:>10} {100*p_sol:>8.3f}% {100*p_per:>10.2e}% "
		      f"{(p_sol/p_per if hit else 0):>10.2e}")
