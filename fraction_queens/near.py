"""A que racional simple se pega la fraccion de una solucion algebraica."""
import sys, os, math
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frac


def convergents(a):
	p0, q0, p1, q1 = 1, 0, a[0], 1
	out = [(p1, q1)]
	for t in a[1:]:
		p0, q0, p1, q1 = p1, q1, t * p1 + p0, t * q1 + q0
		out.append((p1, q1))
	return out


def best_rational(sigma, n):
	"""Convergente justo antes del cociente parcial mas grande."""
	p, q = frac.reduced(sigma, n)
	a = frac.cont_frac(p, q)
	if len(a) < 2:
		return None
	i = max(range(1, len(a)), key=lambda j: a[j])
	cv = convergents(a)[i - 1]
	err = abs(Fraction(p, q) - Fraction(cv[0], cv[1]))     # exacto, sin floats
	return a[i], cv, err


if __name__ == '__main__':
	for n in range(7, 14):
		sols = frac.solutions(n)
		rank = []
		for s in sols:
			r = best_rational(s, n)
			if r:
				rank.append((r[0], s, r[1], r[2]))
		rank.sort(reverse=True, key=lambda t: t[0])
		print(f"=== N={n}  (Q={len(sols)}) ===")
		for A, s, (p, q), err in rank[:4]:
			steps = sorted({(s[k + 1] - s[k]) % n for k in range(n - 1)})
			print(f"  sigma={list(s)}  pasos={steps}")
			print(f"     x ~ {p}/{q}   |x-p/q|={err:.3e}   cociente parcial={A}")
		print()
