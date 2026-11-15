"""
Forma cerrada de kappa_[3]. El objetivo del ejercicio: kappa_[1] = (2N-1)/3 es
exacto y elemental, y `delsarte.py` no tenia el equivalente para el nivel 3.

De la recurrencia:  kappa_[3]/6 = G3 - G1*G2 + G1^3/3.

G3 lleva denominador (N)_6, G2 lleva (N)_4, y G1 = (2N-1)/3 ya viene reducido,
asi que G1^3/3 solo aporta un 27. El minimo comun multiplo es

	Q(N) = 27 * N(N-1)(N-2)(N-3)(N-4)(N-5)

y como kappa_[3] ~ c*N, el numerador v(N) = kappa_[3]*Q(N) deberia tener grado 7.
Verificado: v(N) es ENTERO para todo N medido.

Pero v NO es un polinomio. Sus diferencias finitas no mueren y alternan de
signo: es una **cuasi-polinomica de periodo 2**. Hay que ajustar N par y N impar
por separado. (No sorprende en este problema: el README de diag_waves ya tenia
`S(pi) = 0 para todo N par` y la condicion `gcd(N,6)=1` de la familia mod-N.)

Metodo: ajustar cada clase de paridad con los N MAS GRANDES y validar hacia
abajo. Al reves no vale — los N pequenos estan en regimen de borde, donde los
max(0,...) de la inclusion-exclusion aun muerden, y contaminan el ajuste.
"""

from fractions import Fraction


def Q(n):
	return 27 * n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)


def _fit(pts, deg):
	"""Interpola P de grado deg por los deg+1 puntos (n, y). Racionales."""
	if len(pts) < deg + 1:
		return None
	A = [[Fraction(n) ** j for j in range(deg + 1)] + [Fraction(y)] for n, y in pts]
	m = deg + 1
	for c in range(m):
		piv = next((r for r in range(c, m) if A[r][c]), None)
		if piv is None:
			return None
		A[c], A[piv] = A[piv], A[c]
		pv = A[c][c]
		A[c] = [x / pv for x in A[c]]
		for r in range(m):
			if r != c and A[r][c]:
				f = A[r][c]
				A[r] = [x - f * y for x, y in zip(A[r], A[c])]
	return [A[r][m] for r in range(m)]


def _ev(P, n):
	s = Fraction(0)
	for c in reversed(P):
		s = s * n + c
	return s


def report(data, deg=7):
	"""data: {N: kappa_[3] exacto como Fraction}."""
	v = {n: Fraction(k) * Q(n) for n, k in data.items()}
	assert all(x.denominator == 1 for x in v.values()), "v(N) no es entero"
	res = {}
	for par, name in ((0, "N par  "), (1, "N impar")):
		ns = sorted(n for n in v if n % 2 == par)
		if len(ns) < deg + 2:
			print(f"{name}: {len(ns)} puntos, hacen falta {deg+2} para ajustar y validar")
			continue
		use = ns[-(deg + 1):]
		rest = [n for n in ns if n not in use]
		P = _fit([(n, v[n]) for n in use], deg)
		bad = [n for n in rest if _ev(P, n) != v[n]]
		print(f"{name}: ajuste grado {deg} con N={use[0]}..{use[-1]}, "
		      f"validacion en N={rest}")
		print(f"         -> {'VALIDA en los ' + str(len(rest)) + ' puntos' if not bad else 'falla en ' + str(bad)}")
		if not bad:
			res[par] = P
			print(f"         coeficiente lider: {P[deg]}  = {float(P[deg]):.6f}")
	if len(res) == 2:
		lead = [res[0][deg], res[1][deg]]
		print()
		if lead[0] == lead[1]:
			c = Fraction(lead[0], 27)
			print(f"  mismo termino lider en las dos paridades: la paridad es un efecto")
			print(f"  subdominante. kappa_[3]/N -> {lead[0]}/27 = {float(c):.8f}")
			print(f"  t3/N = -kappa_[3]/(6N) -> {float(-c/6):.8f}")
		else:
			print(f"  lideres distintos por paridad: {lead[0]} vs {lead[1]}")
	return res


if __name__ == "__main__":
	import sys
	from .cumulants import cumulants
	nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 22
	data = {}
	for n in range(6, nmax + 1):
		data[n] = cumulants(n)[3] * 6
		print(f"  N={n}", flush=True)
	report(data)
