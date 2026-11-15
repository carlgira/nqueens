"""
COMPLEJIDAD RACIONAL de la fraccion.

x(sigma) tiene N digitos en base N: hacen falta ~N log N bits para escribirlo.
Pero quiza un racional p/q con q pequeno ya reproduce esos N digitos. Definimos

    q*(sigma) = menor q tal que existe p con |x - p/q| < N^-N
    kappa     = log_N(q*) / N      en [0,1]

Un racional cualquiera de denominador N^N necesita q* ~ N^{N/2} (teoria de
aproximacion diofantica: el mejor convergente da error ~1/q^2), o sea
kappa ~ 0.5. Un numero "comprimible" tiene kappa mucho menor.
"""
import sys, os, math
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import frac


def q_star(sigma, n):
	"""Menor denominador de convergente que reproduce los N digitos."""
	p, q = frac.reduced(sigma, n)
	a = frac.cont_frac(p, q)
	tol = Fraction(1, n ** n)
	x = Fraction(p, q)
	p0, q0, p1, q1 = 1, 0, a[0], 1
	if abs(x - Fraction(p1, q1)) < tol:
		return q1
	for t in a[1:]:
		p0, q0, p1, q1 = p1, q1, t * p1 + p0, t * q1 + q0
		if abs(x - Fraction(p1, q1)) < tol:
			return q1
	return q


def kappa(sigma, n):
	return math.log(q_star(sigma, n)) / (n * math.log(n))


def steps(s, n):
	return sorted({(s[k + 1] - s[k]) % n for k in range(n - 1)})


if __name__ == '__main__':
	print(f"{'N':>3} {'kappa_sol':>10} {'kappa_per':>10} {'min_sol':>8} "
	      f"{'min_per':>8} {'%sol<0.35':>10} {'%per<0.35':>10}")
	for n in range(6, 13):
		sols = frac.solutions(n)
		perms = (frac.permutations_all(n) if n <= 8
		         else frac.permutations_sample(n, max(2000, len(sols)), seed=1))
		ks = np.array([kappa(s, n) for s in sols])
		kp = np.array([kappa(p, n) for p in perms])
		print(f"{n:>3} {ks.mean():>10.4f} {kp.mean():>10.4f} {ks.min():>8.4f} "
		      f"{kp.min():>8.4f} {100*(ks<.35).mean():>10.2f} "
		      f"{100*(kp<.35).mean():>10.2f}")

	print("\n--- las soluciones mas comprimibles de cada N ---")
	for n in range(7, 14):
		sols = frac.solutions(n)
		r = sorted(((kappa(s, n), s) for s in sols))[:3]
		for k, s in r:
			print(f"N={n:>2} kappa={k:.4f} q*={q_star(s, n):>8} "
			      f"pasos={steps(s, n)}  {list(s)}")
		print()
