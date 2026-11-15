r"""
MEZCLAR SOLUCIONES DENTRO DE UN MISMO FRACTAL.

Dos ejes independientes:

  (a) PROFUNDIDAD  -- sustitucion heterogenea de Moran:
        P_1 = tau_1  (cualquier solucion),  P_{k+1} = P_k (x) tau_{k+1}
      con tau_2..tau_k toroidales, de TAMANOS DISTINTOS. Cada nivel de zoom
      guarda una solucion distinta.

  (b) RAMA -- cada bloque i del nivel exterior usa su propia tau^(i).
      Aqui la cosa se pone interesante: no vale cualquier combinacion.
      Con  rho(i*m+j) = sigma(i)*m + tau^(i)(j):

        x - rho(x) = A_i*m + B,   A_i = i - sigma(i),   B = j - tau^(i)(j)

      colisiona si  A_i - A_i' = +-1  y  B - B' = -+m  con B en D_i, B' en D_i'.
      Dentro de un bloque lo prohibe la toroidalidad. ENTRE bloques distintos
      hay que pedirlo a mano:

      REGLA DE COMPATIBILIDAD: tau y tau' son mezclables si D_tau = D_tau' y
      S_tau = S_tau' (mismos conjuntos diagonales), y solo hace falta exigirlo
      a los pares de bloques cuyas diagonales exteriores son ADYACENTES.
"""
import numpy as np
from itertools import permutations
from tensor import valid, toroidal, solutions


def kron_mixed(sig, taus):
	"""taus[i] = solucion usada dentro del bloque i."""
	sig = np.asarray(sig); m = len(taus[0])
	return np.array([sig[i] * m + taus[i][j] for i in range(len(sig)) for j in range(m)])


def diag_sets(t):
	t = np.asarray(t); k = np.arange(len(t))
	return frozenset((k - t).tolist()), frozenset((k + t).tolist())


def predict_mixed(sig, taus):
	"""Predicado exacto derivado a mano."""
	sig = np.asarray(sig); n = len(sig); m = len(taus[0]); k = np.arange(m)
	if not all(valid(t) for t in taus):
		return False
	A = np.arange(n) - sig; C = np.arange(n) + sig
	D = [set((k - np.asarray(t)).tolist()) for t in taus]
	S = [set((k + np.asarray(t)).tolist()) for t in taus]
	for i in range(n):
		for ip in range(n):
			if abs(A[i] - A[ip]) == 1 and any((d - m) in D[ip] for d in D[i]):
				return False
			if abs(C[i] - C[ip]) == 1 and any((s - m) in S[ip] for s in S[ip == ip and i] ) and False:
				pass
			if abs(C[i] - C[ip]) == 1 and any((s - m) in S[ip] for s in S[i]):
				return False
	return valid(sig)
