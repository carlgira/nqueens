"""
El polinomio de conflictos y sus CEROS. Por que la escalera de niveles se para.

delsarte.py dejo el mapa: nivel 1 (marginales) 0%, nivel 2 (pares) 70.6%, y un
30% que "necesita el desarrollo completo de cumulos". Esto responde si ese
desarrollo existe.

El objeto:

	C_N(z) = sum_{sigma en S_N} z^{c(sigma)}        c = pares en conflicto

	C_N(0) = Q(N)        C_N(1) = N!

Es una funcion de particion con z de fugacidad, y Q(N) es su limite de
temperatura cero. Todo el problema es ir de z=1 (donde sabemos contar) a z=0
(donde queremos contar).

El desarrollo de cumulos es exactamente el desarrollo de Taylor de log C_N en
z=1:

	log C_N(z)/N! = sum_k kappa_[k] (z-1)^k / k!       (cumulos factoriales)

	log Q(N)/N!   = sum_k kappa_[k] (-1)^k / k!        evaluado en z-1 = -1

y el nivel 2 del README es el primer termino solo: kappa_[1] = E[C] = (2N-1)/3.
El radio de convergencia de esa serie es la distancia de z=1 al cero mas
cercano de C_N. Medido: R ~ 4.9/N. Se cierra sobre z=1, o sea que z=0 queda
FUERA para todo N>=6 y la serie diverge. No es que falte un termino: la serie
no converge.

Reparametrizacion util. Si r es un cero, poniendo p = 1/(1-r):

	log Q(N)/N! = sum_i log|1 - p_i|          y      sum_i p_i = E[C]  (exacto)

Si todas las p fueran reales en (0,1), C seria una suma de Bernoullis
independientes. Como Var[C] > E[C] (medido en el README, 1.026), eso es
imposible: la sobredispersion PRUEBA que C_N no tiene todas sus raices reales.
Las p complejas son la "correlacion entre pares" del README, en forma cerrada.
"""

import numpy as np
from fractions import Fraction
from math import comb, lgamma, log

QN = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596]


# ---------------------------------------------------------------- enumeracion

def _all_perms(n):
	P = np.zeros((1, 0), dtype=np.int8)
	for k in range(n):
		m = P.shape[0]
		out = np.empty((m * (k + 1), k + 1), dtype=np.int8)
		for pos in range(k + 1):
			b = out[pos * m:(pos + 1) * m]
			b[:, :pos] = P[:, :pos]
			b[:, pos] = k
			b[:, pos + 1:] = P[:, pos:]
		P = out
	return P


def _count(B, n, idx):
	"""c(sigma) por bloque. Ordenando d y contando coincidencias a distancia s:
	una diagonal con m reinas aporta sum_s (m-s)+ = C(m,2)."""
	c = np.zeros(B.shape[0], dtype=np.int32)
	for sg in (1, -1):
		d = np.sort(B + sg * idx, axis=1)
		for s in range(1, n):
			c += (d[:, s:] == d[:, :-s]).sum(axis=1, dtype=np.int32)
	return np.bincount(c, minlength=n * (n - 1) // 2 + 1)


def conflict_hist(n):
	"""Coeficientes EXACTOS de C_N(z). Bloques por el primer valor: memoria
	(n-1)! en vez de n!. N=12 (479M permutaciones) en ~3.5 min."""
	base = _all_perms(n - 1).astype(np.int16)
	idx = np.arange(n, dtype=np.int16)
	hist = np.zeros(n * (n - 1) // 2 + 1, dtype=np.int64)
	B = np.empty((base.shape[0], n), dtype=np.int16)
	for v in range(n):
		B[:, 0] = v
		B[:, 1:] = base + (base >= v)       # reetiquetado saltando v
		hist += _count(B, n, idx)
	return hist


# ------------------------------------------------------------------- cumulos

def fact_cumulants(hist, K):
	"""kappa_[k]/k! exactos en racionales, por la recurrencia L' G = G'."""
	h = [int(x) for x in hist]
	tot = sum(h)
	G = [Fraction(0)] * (K + 1)
	for c, cnt in enumerate(h):
		if not cnt:
			continue
		for k in range(min(c, K) + 1):
			G[k] += Fraction(cnt * comb(c, k), tot)
	L = [Fraction(0)] * (K + 1)
	for k in range(1, K + 1):
		s = G[k] * k
		for j in range(1, k):
			s -= j * L[j] * G[k - j]
		L[k] = s / k
	return L


# --------------------------------------------------------------------- ceros

def roots(hist):
	D = int(np.nonzero(hist)[0][-1])
	co = hist[:D + 1].astype(float) / hist.sum()
	return np.roots(co[::-1])


def report(ns=range(5, 13), cache=None):
	H = {}
	for n in ns:
		H[n] = conflict_hist(n)
		print(f"  ...N={n} listo, C(0)={H[n][0]}", flush=True)
	if cache:
		np.savez(cache, **{str(n): h for n, h in H.items()})

	print("\n== ceros y radio de convergencia ==")
	print(" N  grado  min|r|  min|r-1|=R   N*R   z=0 respecto al disco")
	for n in ns:
		r = roots(H[n])
		R = np.abs(r - 1).min()
		D = len(r)
		print(f"{n:2d} {D:6d} {np.abs(r).min():7.4f} {R:10.4f} {n*R:7.3f}   "
		      f"{'DENTRO' if R > 1 else 'FUERA'} (margen {R-1:+.4f})")

	print("\n== probabilidades de conflicto efectivas p = 1/(1-r) ==")
	print(" N   sum p      E[C]=(2N-1)/3   sum|Im p|   raices reales")
	for n in ns:
		r = roots(H[n])
		p = 1.0 / (1.0 - r)
		print(f"{n:2d} {p.sum().real:10.5f} {(2*n-1)/3:14.5f} {np.abs(p.imag).sum():11.4f}"
		      f" {int(np.sum(np.abs(r.imag) < 1e-8)):13d}/{len(r)}")

	print("\n== la escalera: truncar la serie de cumulos en k ==")
	print(" N     t1      t2      t3      t4      t5  |  captura k=1..5 (%)      t3/N")
	for n in ns:
		L = fact_cumulants(H[n], 6)
		t = [float(L[k]) * (-1) ** k for k in range(1, 7)]
		P = np.cumsum(t)
		true = log(QN[n - 1]) - lgamma(n + 1)
		c = " ".join(f"{100*P[k]/true:6.1f}" for k in range(5))
		print(f"{n:2d} " + " ".join(f"{x:7.3f}" for x in t[:5]) + f"  | {c}  {t[2]/n:8.4f}")

	print("\n== la serie diverge: terminos y sumas parciales, N maximo ==")
	n = max(ns)
	L = fact_cumulants(H[n], 16)
	t = [float(L[k]) * (-1) ** k for k in range(1, 17)]
	P = np.cumsum(t)
	true = log(QN[n - 1]) - lgamma(n + 1)
	print(f"  objetivo log(Q/N!) = {true:.4f}")
	print("   k     termino      parcial   |t_k|^(1/k) -> 1/R")
	for k in range(1, 17):
		print(f" {k:3d} {t[k-1]:13.3f} {P[k-1]:12.3f} {abs(t[k-1])**(1.0/k):12.4f}")
	return H


if __name__ == "__main__":
	import sys
	nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 11
	report(range(5, nmax + 1))
