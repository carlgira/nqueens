"""
Torre tensorial: sigma (x) tau  ->  solucion de tamano n*m.
Hipotesis: basta con que tau sea solucion TOROIDAL para que el producto valga.
"""
import numpy as np
from itertools import permutations


def valid(p):
	p = np.asarray(p); n = len(p); i = np.arange(n)
	return len(set((i + p).tolist())) == n and len(set((i - p).tolist())) == n


def toroidal(p):
	p = np.asarray(p); n = len(p); i = np.arange(n)
	return len(set(((i + p) % n).tolist())) == n and len(set(((i - p) % n).tolist())) == n


def kron(sig, tau):
	"""Bloque: la casilla (i,sig(i)) se sustituye por una copia de tau."""
	sig = np.asarray(sig); tau = np.asarray(tau); m = len(tau)
	return np.array([sig[i] * m + tau[j] for i in range(len(sig)) for j in range(m)])


def solutions(n, limit=None):
	out = []
	for p in permutations(range(n)):
		if valid(p):
			out.append(np.array(p))
			if limit and len(out) >= limit:
				break
	return out
