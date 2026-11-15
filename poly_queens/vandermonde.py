"""
La factorizacion de Vandermonde: N-reinas es la NO ANULACION de un polinomio.

	prod_{i<j} [ (s_i - s_j)^2 - (i - j)^2 ]
	  = prod_{i<j} [ (s_i - i) - (s_j - j) ] * prod_{i<j} [ (s_i + i) - (s_j + j) ]
	  = V(d-) * V(d+)

con d+(i) = s_i + i, d-(i) = s_i - i los dos canales diagonales de siempre.

DEMOSTRACION, entera. Es una diferencia de cuadrados:

	(s_i - s_j)^2 - (i - j)^2 = [(s_i - s_j) - (i - j)] * [(s_i - s_j) + (i - j)]
	                          = [(s_i - i) - (s_j - j)] * [(s_i + i) - (s_j + j)]

y al multiplicar sobre i<j el primer factor recorre exactamente el Vandermonde
de d- y el segundo el de d+. No hay induccion ni caso base: es una identidad de
polinomios en las indeterminadas s_1..s_n, valida para TODO n y para cualquier
vector s -- entero, racional o complejo, permutacion o no. `check_identity`
solo la re-verifica numericamente; no la sostiene.

Lo que si esta limitado por fuerza bruta es la otra mitad, `check_count`: que
los no-nulos son exactamente Q(N). Eso pide recorrer las N! permutaciones. En
este modulo llega a N=9 porque cuesta N!*O(N^2) con enteros grandes; por la via
equivalente de contar conflictos (`conflict_poly.conflict_hist`, O(N) por
permutacion) esta verificado hasta N=12. Mas alla solo se comprueba la
EQUIVALENCIA delta != 0 <=> c = 0, que es lo que hace `check_equiv` (hasta
N=121), no el conteo.

O sea:

	sigma es solucion  <=>  V(sigma + id) * V(sigma - id) != 0

Q(N) = #{sigma en S_N : ese producto no se anula}. Es un enunciado de geometria
de polinomios, no de combinatoria: "cuantos puntos de la orbita de S_N sobre
(1..N) evitan la union de dos discriminantes trasladados".

Dos consecuencias que se usan mas abajo:

1. Es un TESTIGO EXACTO nuevo, distinto del de diag_waves: aquel contaba
   conflictos (<R^2> - N/2), este los detecta por anulacion. Su logaritmo
   log|V(d+)| + log|V(d-)| es MENOS la energia de Coulomb de dos gases 1D:
   las soluciones son los puntos donde ambos gases se repelen sin colisionar.
   Al contrario que las energias cuadraticas de newton_queens/ y wave_queens/,
   esta si tiene barrera infinita en el conflicto y el estado difuminado vale
   -infinito. Es un potencial donde la solucion es un maximo, no un vertice.

2. El metodo polinomico (Nullstellensatz combinatorio de Alon) pide
   grado = sum_i k_i sobre la rejilla. Aqui:

	   deg V(x) V(x+id) V(x-id) = 3*N(N-1)/2      rejilla [N]^N: sum k_i = N(N-1)

   Sobra un factor 3/2. Dicho de otra forma: la rejilla paga DOS familias de
   transversales y el problema tiene TRES (columnas, d+, d-). Es el mismo
   umbral "2 facil, 3 dificil" que separa emparejamiento bipartito de
   3-dimensional matching.
"""

import itertools
import random
from fractions import Fraction
from math import gcd


def V(v):
	"""Vandermonde prod_{i<j} (v_j - v_i), exacto en enteros."""
	p = 1
	for i in range(len(v)):
		for j in range(i + 1, len(v)):
			p *= v[j] - v[i]
	return p


def delta(s):
	"""V(sigma+id) * V(sigma-id). Cero <=> sigma no es solucion."""
	n = len(s)
	return V([s[i] + i for i in range(n)]) * V([s[i] - i for i in range(n)])


def _lhs(s):
	n = len(s)
	p = 1
	for i in range(n):
		for j in range(i + 1, n):
			p *= (s[i] - s[j]) ** 2 - (i - j) ** 2
	return p


def conflicts(s):
	n = len(s)
	return sum(1 for i in range(n) for j in range(i + 1, n)
	           if s[i] + i == s[j] + j or s[i] - i == s[j] - j)


def check_identity(ns=(10, 20, 40, 80, 160), reps=12, seed=0):
	"""La identidad como IDENTIDAD: puntos enteros y racionales cualesquiera,
	nada de permutaciones. Es lo que de verdad hay que comprobar."""
	rnd = random.Random(seed)
	print("identidad en puntos arbitrarios")
	print("    N   pruebas   discrepancias   bits de |V*V|")
	for n in ns:
		bad = 0
		bits = 0
		for _ in range(reps):
			s = [rnd.randint(-10 ** 6, 10 ** 6) for _ in range(n)]
			r = delta(s)
			bad += (_lhs(s) != r)
			bits = max(bits, abs(r).bit_length())
		print(f"  {n:4d} {reps:9d} {bad:14d} {bits:16d}")
	bad = 0
	for _ in range(200):
		n = rnd.randint(2, 12)
		s = [Fraction(rnd.randint(-500, 500), rnd.randint(1, 97)) for _ in range(n)]
		bad += (_lhs(s) != delta(s))
	print(f"  200 vectores racionales, N=2..12 -> discrepancias: {bad}")


def check_equiv(ns=(13, 25, 35, 49, 55, 77, 91, 121), reps=60, seed=1):
	"""delta != 0 <=> c = 0, lejos del alcance de la enumeracion. Las
	soluciones vienen de la familia (a*i+b) mod N, que pide gcd(N,6)=1."""
	rnd = random.Random(seed)
	print("\nequivalencia  delta(sigma) != 0  <=>  c(sigma) = 0")
	print("   N   perms al azar   soluciones mod-N   discrepancias")
	for n in ns:
		bad = 0
		for _ in range(reps):
			s = list(range(n))
			rnd.shuffle(s)
			bad += (delta(s) != 0) != (conflicts(s) == 0)
		sols = 0
		for a in range(1, n):
			if gcd(a, n) != 1:
				continue
			for b in range(n):
				s = [(a * i + b) % n for i in range(n)]
				if conflicts(s) == 0:
					sols += 1
					bad += (delta(s) == 0)
				if sols >= 40:
					break
			if sols >= 40:
				break
		print(f" {n:4d} {reps:15d} {sols:18d} {bad:15d}")


def check_count(nmax=9):
	"""Los no-nulos son exactamente Q(N). Fuerza bruta: N!*O(N^2) con enteros
	grandes, de ahi el techo. Para N=10..12 usar conflict_poly.conflict_hist."""
	print("\nconteo exhaustivo")
	print(" N   no-nulos   Q(N)")
	QN = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200]
	for n in range(3, nmax + 1):
		nz = sum(1 for s in itertools.permutations(range(n)) if delta(s))
		print(f"{n:2d} {nz:10d} {QN[n-1]:6d}   {'ok' if nz == QN[n-1] else 'FALLA'}")


if __name__ == "__main__":
	check_identity()
	check_equiv()
	check_count()
