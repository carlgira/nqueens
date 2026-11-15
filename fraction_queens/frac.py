"""
LA REPRESENTACION FRACCIONAL DE UNA PERMUTACION.

Una permutacion sigma de {0..N-1} se lee como los digitos en base N de un
numero de [0,1):

    x(sigma) = sum_k sigma(k) * N^{-(k+1)}   =  M / N^N ,  M entero.

Es una biyeccion permutaciones <-> ciertos racionales de denominador N^N.
La pregunta: los x de las SOLUCIONES de N-reinas, como numeros, se distinguen
de los x de las permutaciones cualesquiera?
"""
import math
from functools import lru_cache
import random

# ---------------------------------------------------------------- soluciones

def solutions(n):
	"""Todas las soluciones como tuplas sigma (columna por fila)."""
	out = []
	col = [False] * n
	d1 = [False] * (2 * n)
	d2 = [False] * (2 * n)
	cur = [0] * n

	def go(k):
		if k == n:
			out.append(tuple(cur))
			return
		for c in range(n):
			if col[c] or d1[k + c] or d2[k - c + n]:
				continue
			col[c] = d1[k + c] = d2[k - c + n] = True
			cur[k] = c
			go(k + 1)
			col[c] = d1[k + c] = d2[k - c + n] = False
	go(0)
	return out


def prefixes(n, depth):
	"""Prefijos parciales validos de longitud `depth` (nodos vivos del arbol)."""
	out = []
	col = [False] * n
	d1 = [False] * (2 * n)
	d2 = [False] * (2 * n)
	cur = [0] * n

	def go(k):
		if k == depth:
			out.append(tuple(cur[:depth]))
			return
		for c in range(n):
			if col[c] or d1[k + c] or d2[k - c + n]:
				continue
			col[c] = d1[k + c] = d2[k - c + n] = True
			cur[k] = c
			go(k + 1)
			col[c] = d1[k + c] = d2[k - c + n] = False
	go(0)
	return out


def permutations_all(n):
	import itertools
	return list(itertools.permutations(range(n)))


def permutations_sample(n, m, seed=0):
	rng = random.Random(seed)
	base = list(range(n))
	out = []
	seen = set()
	while len(out) < m:
		rng.shuffle(base)
		t = tuple(base)
		if t not in seen:
			seen.add(t)
			out.append(t)
	return out

# ------------------------------------------------------------- la fraccion

def numerator(sigma, n=None):
	"""M entero tal que x = M / N^N."""
	n = n or len(sigma)
	m = 0
	for d in sigma:
		m = m * n + d
	return m


def frac(sigma, n=None):
	"""x en [0,1) como float."""
	n = n or len(sigma)
	x = 0.0
	for d in reversed(sigma):
		x = (x + d) / n
	return x


def reduced(sigma, n=None):
	"""(p, q) con x = p/q irreducible."""
	n = n or len(sigma)
	m = numerator(sigma, n)
	q = n ** n
	g = math.gcd(m, q)
	return m // g, q // g


def cont_frac(p, q, limit=10 ** 6):
	"""Fraccion continua de p/q (racional -> lista finita)."""
	a = []
	while q and len(a) < limit:
		a.append(p // q)
		p, q = q, p - (p // q) * q
	return a
