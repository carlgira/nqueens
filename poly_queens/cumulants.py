"""
Cumulos factoriales de C(sigma) SIN enumerar permutaciones.

conflict_poly.py necesita las N! permutaciones y muere en N=12. Pero para la
escalera solo hacen falta los primeros cumulos, y esos se pueden contar por
otro lado.

Un conflicto es una restriccion LINEAL entre dos filas:

	d+ :  i + sigma(i) = j + sigma(j)   <=>   sigma(i) - sigma(j) = +(j-i)
	d- :  sigma(i) - i = sigma(j) - j   <=>   sigma(i) - sigma(j) = -(j-i)

Hay M = N(N-1) de estas restricciones ("atomos"). C = numero de atomos que se
cumplen, y el momento factorial k-esimo es una suma sobre k-subconjuntos:

	E[(C)_k] / k!  =  sum_{|S|=k} P(S)

Para un S dado, las restricciones forman un grafo sobre las filas. Cada
componente conexa queda determinada por UN valor libre t_p: sigma(r) = t_p +
o_p(r). Entonces

	P(S) = D(S) * (N - n)! / N!

con n el total de filas tocadas y D(S) el numero de tuplas (t_1..t_m) que
mantienen todos los valores dentro de [0,N) y todos distintos. D(S) se cuenta
por inclusion-exclusion sobre las colisiones entre componentes; cada colision
fija una diferencia t_p - t_q, y entonces la tupla vive en una interseccion de
intervalos.

Control de k=1, a mano:  un atomo de hueco g da D = N-g, n = 2, luego
sum_S P = 2 * sum_g (N-g)^2 / (N(N-1)) = (2N-1)/3.  Es el E[C] del README.

Coste: C(M,k). Para k=3 eso es ~N^6/6, o sea N~20-24 en vez de N=12. Todo en
racionales exactos, y verificado contra conflict_poly para N<=12.
"""

import itertools
from fractions import Fraction
from math import factorial


def atoms(n):
	"""(i, j, d) con i<j y la restriccion sigma[i] - sigma[j] = d."""
	out = []
	for i in range(n):
		for j in range(i + 1, n):
			g = j - i
			out.append((i, j, g))    # d+
			out.append((i, j, -g))   # d-
	return out


def _components(subset):
	"""Union-find con potencial sobre las filas. Devuelve la lista de
	componentes como conjuntos de offsets normalizados (min 0), o None si el
	sistema es inconsistente o pide dos filas con el mismo valor."""
	rows = sorted({r for a in subset for r in a[:2]})
	idx = {r: k for k, r in enumerate(rows)}
	par = list(range(len(rows)))
	off = [0] * len(rows)                       # val(x) = val(par[x]) + off[x]

	def find(x):
		o = 0
		while par[x] != x:
			o += off[x]
			x = par[x]
		return x, o

	for i, j, d in subset:
		a, oa = find(idx[i])
		b, ob = find(idx[j])
		if a == b:
			if oa - ob != d:
				return None                     # ciclo inconsistente
		else:
			par[a] = b
			off[a] = d - oa + ob                # val(a) = val(b) + (...)

	groups = {}
	for r in rows:
		root, o = find(idx[r])
		groups.setdefault(root, []).append(o)
	comps = []
	for os in groups.values():
		if len(set(os)) != len(os):
			return None                         # dos filas con el mismo valor
		m = min(os)
		comps.append(sorted(o - m for o in os))
	return comps


def _count_exact(Ls, sel, combo):
	"""Tuplas (t_p) con las diferencias t_p - t_q EXACTAMENTE las de combo."""
	m = len(Ls)
	par = list(range(m))
	off = [0] * m

	def find(x):
		o = 0
		while par[x] != x:
			o += off[x]
			x = par[x]
		return x, o

	for (p, q), d in zip(sel, combo):
		a, oa = find(p)
		b, ob = find(q)
		if a == b:
			if oa - ob != d:
				return 0
		else:
			par[a] = b
			off[a] = d - oa + ob

	groups = {}
	for p in range(m):
		root, o = find(p)
		groups.setdefault(root, []).append((p, o))
	total = 1
	for members in groups.values():
		# t_p = t_root + o  en [0, L_p)  ->  t_root en [-o, L_p - o)
		lo = max(-o for _, o in members)
		hi = min(Ls[p] - o for p, o in members)
		if hi <= lo:
			return 0
		total *= hi - lo
	return total


def _count_tuples(Ls, Os):
	"""Tuplas con todos los valores distintos entre componentes, por
	inclusion-exclusion sobre los pares que colisionan."""
	m = len(Ls)
	if m == 1:
		return Ls[0]
	pairs = [(p, q) for p in range(m) for q in range(p + 1, m)]
	# colision entre p y q  <=>  t_p - t_q = o_q - o_p para algun par de offsets
	Delta = {pq: sorted({oq - op for op in Os[pq[0]] for oq in Os[pq[1]]})
	         for pq in pairs}
	total = 0
	for r in range(len(pairs) + 1):
		for sel in itertools.combinations(pairs, r):
			sign = -1 if r % 2 else 1
			if not sel:
				t = 1
				for L in Ls:
					t *= L
				total += t
				continue
			for combo in itertools.product(*[Delta[pq] for pq in sel]):
				total += sign * _count_exact(Ls, sel, combo)
	return total


def _key(comps, n):
	"""Firma canonica de un S: D(S) solo depende de los offsets de las
	componentes (y de N). Dos S con la misma firma comparten D, y eso es lo que
	hace viable llegar a N~24: hay O(N^3) firmas frente a O(N^6) subconjuntos."""
	return tuple(sorted((n - c[-1], tuple(c)) for c in comps))


def D_of(comps, n, memo):
	"""D(S) memoizado por firma."""
	k = _key(comps, n)
	d = memo.get(k)
	if d is None:
		Ls = [x[0] for x in k]
		Os = [list(x[1]) for x in k]
		d = 0 if any(L <= 0 for L in Ls) else _count_tuples(Ls, Os)
		memo[k] = d
	return d


def prob(subset, n, fact, memo=None):
	"""P(S) exacta, como Fraction."""
	comps = _components(subset)
	if comps is None:
		return None
	D = D_of(comps, n, {} if memo is None else memo)
	if D <= 0:
		return None
	nrows = sum(len(c) for c in comps)
	return Fraction(D * fact[n - nrows], fact[n])


def G(n, kmax=3, verbose=False):
	"""G[k] = sum_{|S|=k} P(S) = E[(C)_k]/k!, exacto.

	El bucle interno no toca Fraction: acumula D en enteros, separados por el
	numero de filas tocadas, y divide una sola vez al final."""
	A = atoms(n)
	fact = [factorial(i) for i in range(n + 1)]
	out = [Fraction(1)] + [Fraction(0)] * kmax
	memo = {}
	for k in range(1, kmax + 1):
		acc = [0] * (2 * k + 1)                 # acc[nrows] += D
		for S in itertools.combinations(A, k):
			comps = _components(S)
			if comps is None:
				continue
			D = D_of(comps, n, memo)
			if D > 0:
				acc[sum(len(c) for c in comps)] += D
		out[k] = sum(Fraction(acc[r] * fact[n - r], fact[n])
		             for r in range(2, 2 * k + 1) if acc[r])
		if verbose:
			print(f"    k={k}: {len(memo)} firmas distintas de {len(A)} atomos", flush=True)
	return out


def cumulants(n, kmax=3, verbose=False):
	"""kappa_[k]/k! por la misma recurrencia L' G = G' de conflict_poly."""
	g = G(n, kmax, verbose)
	L = [Fraction(0)] * (kmax + 1)
	for k in range(1, kmax + 1):
		s = g[k] * k
		for j in range(1, k):
			s -= j * L[j] * g[k - j]
		L[k] = s / k
	return L


if __name__ == "__main__":
	import sys, time
	from math import gcd
	nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
	print(" N    kappa_[1]    kappa_[2]        kappa_[3]              t3=-k3/6   t3/N      s")
	for n in range(4, nmax + 1):
		t = time.time()
		L = cumulants(n)
		k1, k2, k3 = L[1], L[2] * 2, L[3] * 6
		t3 = -float(k3) / 6
		print(f"{n:3d} {str(k1):>10s} {str(k2):>12s} {str(k3):>18s} {t3:11.4f} {t3/n:9.5f}"
		      f" {time.time()-t:8.1f}", flush=True)
