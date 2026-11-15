r"""
UN SOLO FRACTAL CON TODAS LAS SOLUCIONES.

Torre de Moran heterogenea: P_1 = tau_1 (cualquier solucion),
P_{k+1} = P_k (x) tau_{k+1} con tau_{k+1} TOROIDAL, de tamano libre.

El resultado es una solucion honesta de N = prod(m_k) reinas, y cada
profundidad de zoom guarda una solucion distinta. No es autosemejante: es
autosemejante-por-niveles (conjunto de Moran).
"""
import numpy as np
from tensor import kron, valid, toroidal


def tower(taus):
	P = np.asarray(taus[0])
	for t in taus[1:]:
		P = kron(P, np.asarray(t))
	return P


def window(P, sizes, level, branch=None):
	"""Ventana que aisla el nivel `level` (1-indexado). Devuelve puntos locales."""
	w = int(np.prod(sizes[level - 1:]))
	r0 = 0
	if branch:
		off = 0
		for k, j in enumerate(branch[:level - 1]):
			off += j * int(np.prod(sizes[k + 1:]))
		r0 = off
	rows = np.arange(r0, r0 + w)
	cols = P[rows]
	c0 = cols.min() - cols.min() % w
	return rows - r0, cols - c0, w


def spiral_maps(sigma, dim=1.85):
	sigma = np.asarray(sigma); m = len(sigma); M = 2 * m - 1; k = np.arange(m)
	r = m ** (-1.0 / dim)
	return r * np.exp(2j * np.pi * (k + sigma) / M), np.exp(2j * np.pi * (k - sigma + m - 1) / M)


def moran_points(taus, dim=1.85):
	"""Atractor de la torre heterogenea: un nivel de IFS por solucion."""
	z = np.zeros(1, complex)
	for t in reversed(taus):
		lam, tr = spiral_maps(t, dim)
		z = (lam[:, None] * z[None, :] + tr[:, None]).ravel()
	return z
