"""
Los CUELLOS DE BOTELLA: los valores discretos por los que pasa S(x) en los
puntos racionales de pi, y si sirven para recortar el espacio.

En x = 2*pi*j/m la onda colapsa a un conteo modular y solo puede caer en una
reticula finita de valores -- los "cuellos". Tres preguntas, tres respuestas.

1) LAS 8 SOLUCIONES DE UNA ORBITA, MISMO CUELLO O DISTINTOS?

   Depende del punto, y la condicion es exacta:

	   S(x) es invariante del grupo diedrico  <=>  2N*x = 0 (mod 2pi)  <=>  m | 2N

   Sale de que cos((2N-k)x) = cos(kx) en esos puntos, lo que hace
   intercambiables las dos bandas. Verificado en N=7 (m = 2,7,14 concentran) y
   N=8 (m = 2,4,8). Fuera de ahi la orbita se reparte entre varios cuellos.

   Consecuencia: en los puntos que concentran, "quedarse con una de la orbita"
   no da nada -- pasan las 8 o ninguna.

2) HAY UN CUELLO POR EL QUE PASE AL MENOS UNA SOLUCION DE CADA ORBITA?

   Si. Y lo interesante no es que exista, sino POR QUE es barato: las
   soluciones ocupan MUCHOS MENOS cuellos que las permutaciones al azar.

	   N     punto     cuellos usados por las soluciones / por el mismo
						numero de permutaciones al azar
	   8    2pi/8                6 / 31
	  10    2pi/8               15 / 49
	  11    2pi/6               23 / 42
	  12    2pi/8               25 / 68

   Eso es una propiedad real y fuerte: contra un control de 25 grupos al azar
   del mismo tamano, p < 0.001 en N = 9..12.

3) SIRVE PARA PODAR?

   Solo si el soporte se puede DERIVAR sin conocer las soluciones. Y ahi esta
   el problema: los espectros factibles predicen el doble de cuellos de los que
   las soluciones usan de verdad (N=12, x=2pi/6: 22 permitidos, 11 usados).
   Esa mitad sobrante son los fantasmas, otra vez. La regla derivable deja:

	   N        8      9     10     11     12
	   deja  54.4%  38.2%  72.8%  82.9%  91.4%

   Se degrada hasta ser inutil. La version ajustada a las soluciones llega al
   27.9% en N=12, pero no se puede calcular sin resolver el problema.

Y una nota estructural: la forma INCLUYENTE (union de celdas) es siempre peor
que la EXCLUYENTE (interseccion), porque |A union B| >= max(|A|,|B|). La union
solo puede hincharse; la interseccion encoge.
"""

import numpy as np
from math import gcd

from diag_waves.pp import feasible_coefs
from diag_waves.orbits import orbits

GREY, BLUE, ORANGE, PURPLE, GREEN = '#dee2e6', '#1c7ed6', '#e8590c', '#7048e8', '#0ca678'


def coefs(P, n):
	i = np.arange(n)
	c = np.zeros((len(P), 4 * n))
	r = np.repeat(np.arange(len(P)), n)
	np.add.at(c, (r, (i + P).ravel() + 1), 1)
	np.add.at(c, (r, (i - P).ravel() + 3 * n), 1)
	return c


def rational_points(mmax=12):
	return [(m, j) for m in range(2, mmax + 1)
			for j in range(1, m // 2 + 1) if gcd(j, m) == 1]


def concentrates(n, m):
	"""S(2*pi*j/m) es invariante de simetria <=> m divide 2N."""
	return (2 * n) % m == 0


def support(n, mmax=12, nsample=300000, seed=0):
	"""
	Cuantos cuellos usa cada poblacion en cada punto: soluciones, permutaciones
	al azar, y cuantos PERMITEN los espectros factibles (lo unico derivable).
	"""
	rng = np.random.default_rng(seed)
	k = np.arange(4 * n)
	F = feasible_coefs(n)
	O = orbits(n)
	SOL = np.array([p for o in O for p in o])
	CS = coefs(SOL, n)
	PS = np.array([rng.permutation(n) for _ in range(nsample)])
	CP = coefs(PS, n)
	RS = coefs(np.array([rng.permutation(n) for _ in range(len(SOL))]), n)
	rows = []
	for m, j in rational_points(mmax):
		w = np.cos(k * 2 * np.pi * j / m)
		vs = np.round(CS @ w, 9); vp = np.round(CP @ w, 9)
		vf = np.round(F @ w, 9); vr = np.round(RS @ w, 9)
		nv = len(np.unique(vp))
		if nv > 4 * n:
			continue
		allowed = np.unique(vf)
		rows.append(dict(lab=f'2pi*{j}/{m}', m=m, j=j, valores=nv,
						 conc=concentrates(n, m),
						 sols=len(np.unique(vs)), azar=len(np.unique(vr)),
						 permitidos=len(allowed),
						 deja=float(np.isin(vp, allowed).mean()),
						 seguro=bool(np.isin(vs, allowed).all())))
	return rows


def apriori_chain(n, mmax=12, nsample=300000, seed=0):
	"""La regla derivable: S(x) debe caer en un cuello que algun espectro
	factible alcance. Exacta (no pierde soluciones) para todo N."""
	rng = np.random.default_rng(seed)
	k = np.arange(4 * n)
	F = feasible_coefs(n)
	O = orbits(n)
	CS = coefs(np.array([p for o in O for p in o]), n)
	CP = coefs(np.array([rng.permutation(n) for _ in range(nsample)]), n)
	alive = np.ones(nsample, bool)
	for m, j in rational_points(mmax):
		w = np.cos(k * 2 * np.pi * j / m)
		vp = np.round(CP @ w, 9)
		if len(np.unique(vp)) > 4 * n:
			continue
		allowed = np.unique(np.round(F @ w, 9))
		assert np.isin(np.round(CS @ w, 9), allowed).all(), 'perderia soluciones'
		alive &= np.isin(vp, allowed)
	return float(alive.mean())


def plot(path='diag_waves/fig_necks.png'):
	"""Las tres curvas: ajustada, control al azar, y la unica derivable."""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from math import factorial

	ns = [8, 9, 10, 11, 12]
	QN = {8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200}
	ajustada = [.226, 6.341, 7.822, 11.498, 27.914]			# usa el soporte real
	control = [.48, 17.71, 20.99, 37.92, 51.94]				# grupos al azar
	derivable = [54.4, 38.2, 72.8, 82.9, 91.4]				# solo con N
	suelo = [100 * QN[n] / factorial(n) for n in ns]

	fig, ax = plt.subplots(figsize=(8.6, 5.2))
	ax.semilogy(ns, derivable, 'o-', color=PURPLE, lw=2.4,
				label='derivable solo con N (legitima)')
	ax.semilogy(ns, control, '^--', color=GREY, lw=2,
				label='control: grupos al azar')
	ax.semilogy(ns, ajustada, 's-', color=GREEN, lw=2.4,
				label='ajustada al soporte real de las soluciones')
	ax.semilogy(ns, suelo, ':', color=ORANGE, lw=2, label='suelo Q(N)/N!')
	ax.set_xlabel('N'); ax.set_ylabel('% del espacio que sobrevive')
	ax.grid(alpha=.25); ax.legend(fontsize=9)
	ax.set_title('los cuellos: la estructura existe, pero no se puede calcular')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	for n in (8, 10, 12):
		print(f'N={n}')
		for r in support(n):
			print('   {lab:<10} {valores:>3} cuellos  sols {sols:>3}  azar {azar:>3}  '
				  'permitidos {permitidos:>3}  deja {deja:>6.1%}  {c}'.format(
					  c='(concentra la simetria)' if r['conc'] else '', **r))
	plot()
