"""
La geometria DENTRO de una orbita de simetria, y entre orbitas.

Pregunta: las 8 soluciones que forman un grupo de simetria, se ven distintas
en la representacion ondulatoria?

Respuesta: NO, si se mira el modulo. El grupo diedrico actua sobre las dos
diagonales asi (girando 90 grados):

	d+  ->  (N-1) - d-              d-  ->  d+ - (N-1)

es decir: intercambia los dos canales, con una reflexion y un desplazamiento.
Y la trasposicion manda d- -> -d-. Ninguna de esas tres operaciones cambia el
MODULO de la onda de un canal:

	reflexion  d -> M-d   =>  W -> exp(iMx) * conj(W)     |W| igual
	traslacion d -> d-c   =>  W -> exp(-icx) * W          |W| igual
	negacion   d -> -d    =>  W -> conj(W)                |W| igual

Asi que el par NO ORDENADO { |W+(x)|, |W-(x)| } es un invariante exacto de la
orbita: las 8 soluciones dan la MISMA curva. Verificado en N = 7..10, sin una
sola excepcion.

Lo que si varia dentro de la orbita -- S(x), |Z(x)|, M3, M4, <S^4>, hasta 8
valores distintos -- varia por como MONTAMOS las dos bandas en una sola senal
(la banda d+ en los armonicos bajos y la d- en los altos), no por geometria.
Ese montaje rompe la simetria a mano.

Consecuencia practica: todo el embudo de pp.py trata las 8 soluciones de una
orbita como 8 objetos distintos. Trabajando con el invariante se pasa de Q(N)
objetos a U(N) = Q(N)/8.

Y el limite, que es el de siempre: el invariante NO separa orbitas distintas.
En N=10, 92 orbitas dan solo 60 curvas -- 25 chocan. Dos orbitas con la misma
autocorrelacion en los dos canales. Es la ambiguedad de recuperacion de fase
otra vez, ahora al nivel de orbita.
"""

import numpy as np
from collections import defaultdict

from diag_waves.orbits import sols, orbits, lex_index
from diag_waves.diag_waves import channels, shift, occupancy, band_series

GREY, BLUE, ORANGE, PURPLE, GREEN = '#dee2e6', '#1c7ed6', '#e8590c', '#7048e8', '#0ca678'


def grid(m=400):
	return np.linspace(0, 2 * np.pi, m)


def channel_power(p, x):
	"""(|W+(x)|^2, |W-(x)|^2). Cada uno es la transformada de la autocorrelacion
	de su conjunto de diagonales."""
	n = len(p)
	a, b = shift(*channels(p), n)
	k = np.arange(1, 2 * n)
	E = np.exp(1j * np.outer(k, x)).T
	cp = occupancy(a, n)[1:2 * n]
	cm = occupancy(b, n)[1:2 * n]
	return np.abs(E @ cp) ** 2, np.abs(E @ cm) ** 2


def invariant(p, x, dec=6):
	"""El par ordenado canonicamente: invariante exacto de la orbita."""
	P, M = channel_power(p, x)
	a, b = np.round(P, dec), np.round(M, dec)
	return np.vstack([a, b] if tuple(a) <= tuple(b) else [b, a])


def _key(v, dec=6):
	return np.round(v, dec).tobytes()


def check(ns=(7, 8, 9, 10), m=400):
	"""Constante dentro de la orbita? Separa orbitas distintas?"""
	x = grid(m)
	print(f'{"N":>3} {"sols":>6} {"orbitas":>8} {"valores/orbita":>15} {"curvas":>7} {"chocan":>7}')
	for n in ns:
		O = orbits(n)
		per, seen = [], defaultdict(set)
		for oi, o in enumerate(O):
			ks = {_key(invariant(p, x)) for p in o}
			per.append(len(ks))
			for k in ks:
				seen[k].add(oi)
		coll = sum(1 for v in seen.values() if len(v) > 1)
		print(f'{n:>3} {sum(len(o) for o in O):>6} {len(O):>8} '
			  f'{max(per):>15} {len(seen):>7} {coll:>7}')


def orbit_curves(n, m=400):
	"""Una curva por orbita, y el representante lexicografico de cada una."""
	x = grid(m)
	O = orbits(n)
	C = np.array([invariant(o[0], x).ravel() for o in O])
	return O, C, x


def distances(C):
	"""Distancia L2 entre las curvas invariantes, normalizada."""
	d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1) / C.shape[1])
	return d


def hamming(p, q):
	return int(sum(a != b for a, b in zip(p, q)))


def chain(n, seed=None, m=400):
	"""
	Una solucion de cada orbita, encadenadas por cercania.

	La orbita se elige por la distancia entre curvas invariantes (que es la
	unica distancia bien definida entre orbitas). Dentro de la orbita elegida,
	el miembro concreto se elige por distancia de Hamming al anterior -- el
	invariante no puede distinguirlos, asi que hace falta otro criterio.
	"""
	O, C, x = orbit_curves(n, m)
	D = distances(C)
	rng = np.random.default_rng(0)
	s = rng.integers(len(O)) if seed is None else seed
	order = [int(s)]
	left = set(range(len(O))) - {int(s)}
	while left:
		j = min(left, key=lambda t: D[order[-1], t])
		order.append(j); left.discard(j)
	reps = [O[order[0]][0]]
	for j in order[1:]:
		reps.append(min(O[j], key=lambda p: hamming(p, reps[-1])))
	steps = [D[order[i], order[i + 1]] for i in range(len(order) - 1)]
	hams = [hamming(reps[i], reps[i + 1]) for i in range(len(reps) - 1)]
	return order, reps, np.array(steps), np.array(hams), (O, C, D, x)


def _mpl():
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	return plt


def plot_inside(n=8, path=None):
	"""Una orbita entera: el invariante coincide, lo demas no."""
	plt = _mpl()
	path = path or f'diag_waves/fig_orbit_inside_n{n}.png'
	O = orbits(n)
	o = max(O, key=len)
	x = grid(600)
	kk = np.arange(4 * n)
	fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
	cmap = plt.get_cmap('viridis')
	for j, p in enumerate(o):
		col = cmap(j / max(len(o) - 1, 1))
		ax[0].plot(x, band_series(p) @ np.cos(np.outer(kk, x)), color=col, lw=1.2, alpha=.85)
		P, M = channel_power(p, x)
		ax[1].plot(x, P, color=col, lw=2.4, alpha=.5)
		ax[1].plot(x, M, color=col, lw=2.4, alpha=.5, ls='--')
	ax[0].set_title(f'S(x): las {len(o)} soluciones de la orbita, {len(set(_key(band_series(p) @ np.cos(np.outer(kk, x))) for p in o))} curvas distintas')
	ax[1].set_title('|W+|^2 (linea) y |W-|^2 (rayas): las 8 encima de 2 curvas')
	for a in ax:
		a.set_xlabel('x')
	ax[0].set_ylabel('S(x)'); ax[1].set_ylabel('potencia por canal')
	fig.suptitle(f'N={n} — dentro de una orbita de simetria', y=.99)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_between(n=9, path=None):
	"""El mapa de orbitas: MDS sobre la distancia entre curvas invariantes."""
	plt = _mpl()
	path = path or f'diag_waves/fig_orbit_map_n{n}.png'
	order, reps, steps, hams, (O, C, D, x) = chain(n)
	# MDS clasico
	J = np.eye(len(D)) - 1 / len(D)
	B = -.5 * J @ (D ** 2) @ J
	w, V = np.linalg.eigh(B)
	Y = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))

	fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
	sz = np.array([len(o) for o in O])
	ax[0].plot(Y[order, 0], Y[order, 1], '-', color=GREY, lw=1, zorder=1)
	sc = ax[0].scatter(Y[:, 0], Y[:, 1], c=sz, cmap='coolwarm', s=60,
					   edgecolor='k', linewidth=.4, zorder=2)
	ax[0].scatter(Y[order[0], 0], Y[order[0], 1], s=220, facecolor='none',
				  edgecolor=ORANGE, lw=2.5, zorder=3, label='semilla')
	plt.colorbar(sc, ax=ax[0], label='tamano de la orbita')
	ax[0].legend(fontsize=9)
	ax[0].set_title(f'{len(O)} orbitas, MDS sobre la distancia entre invariantes')

	ax[1].plot(steps, 'o-', color=GREEN, lw=1.5, label='salto entre orbitas (curva)')
	ax[1].set_xlabel('paso de la cadena'); ax[1].set_ylabel('distancia L2', color=GREEN)
	a2 = ax[1].twinx()
	a2.plot(hams, 's:', color=ORANGE, lw=1.2, alpha=.8)
	a2.set_ylabel('filas que cambian (Hamming)', color=ORANGE)
	ax[1].set_title(f'la cadena: mediana {np.median(steps):.2f} en curva, '
					f'{np.median(hams):.0f} filas de {n}')
	fig.suptitle(f'N={n} — un representante por orbita, encadenados por cercania', y=.99)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


# ------------------------------------------------------- las orbitas que chocan

def collisions(n, m=400):
	"""
	Grupos de orbitas DISTINTAS que comparten la curva invariante.

	Medido: en todas ellas los dos conjuntos de diagonales son LOS MISMOS.
	13 de 13 pares en N=9; 38 de 40 en N=10 (los otros 2 son una ambiguedad
	de turnpike de verdad). O sea que lo que el invariante no ve no es el
	conjunto de diagonales: es COMO SE EMPAREJAN. La pared de siempre, aislada.
	"""
	x = grid(m)
	O = orbits(n)
	g = defaultdict(list)
	for oi, o in enumerate(O):
		g[invariant(o[0], x).tobytes()].append(oi)
	return O, [v for v in g.values() if len(v) > 1], x


def same_sets(oa, ob):
	"""Busca en las dos orbitas un par con los mismos conjuntos d+ y d-."""
	for p in oa:
		ap, am = set(channels(p)[0]), set(channels(p)[1])
		for q in ob:
			if (ap, am) == (set(channels(q)[0]), set(channels(q)[1])):
				return p, q
	return None


def plot_collisions(n=9, path=None, m=600):
	"""Todas las curvas invariantes juntas; en color las que se solapan."""
	plt = _mpl()
	path = path or f'diag_waves/fig_orbit_collide_n{n}.png'
	O, groups, _ = collisions(n)
	x = grid(m)
	cur = [invariant(o[0], x) for o in O]
	marked = {i for g in groups for i in g}

	fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))
	cmap = plt.get_cmap('tab10')
	for i, c in enumerate(cur):
		if i in marked:
			continue
		ax[0].plot(x, c[0], color=GREY, lw=.8, alpha=.8)
		ax[0].plot(x, c[1], color=GREY, lw=.8, alpha=.8)
	for gi, g in enumerate(groups):
		col = cmap(gi % 10)
		for i in g:
			ax[0].plot(x, cur[i][0], color=col, lw=1.6, alpha=.85)
			ax[0].plot(x, cur[i][1], color=col, lw=1.6, alpha=.85)
	ax[0].set_title(f'{len(O)} orbitas de N={n}: en gris las {len(O)-len(marked)} unicas, '
					f'en color los {len(groups)} grupos que chocan')
	ax[0].set_xlabel('x'); ax[0].set_ylabel('|W|^2 por canal')

	g = groups[0]
	col = cmap(0)
	for j, i in enumerate(g):
		ax[1].plot(x, cur[i][0], color=col if j == 0 else ORANGE,
				   lw=4 if j == 0 else 1.6, alpha=.55 if j == 0 else 1,
				   ls='-' if j == 0 else '--')
		ax[1].plot(x, cur[i][1], color=col if j == 0 else ORANGE,
				   lw=4 if j == 0 else 1.6, alpha=.55 if j == 0 else 1,
				   ls='-' if j == 0 else '--')
	d = float(np.abs(cur[g[0]] - cur[g[1]]).max())
	ax[1].set_title(f'un grupo de cerca: {len(g)} orbitas, diferencia maxima {d:.1e}')
	ax[1].set_xlabel('x')
	fig.suptitle(f'N={n} — orbitas distintas con la MISMA curva invariante', y=.98)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_pair(n=9, which=0, path=None, m=600):
	"""
	Un par que choca. Comparten TODO lo espectral -- el mismo conjunto d+, el
	mismo d-, el mismo espectro c, la misma S(x), la misma curva invariante.
	Lo unico que cambia es el EMPAREJAMIENTO: que d+ va con que d-.
	"""
	plt = _mpl()
	path = path or f'diag_waves/fig_orbit_pair_n{n}.png'
	O, groups, _ = collisions(n)
	g = groups[which]
	pq = same_sets(O[g[0]], O[g[1]]) or (O[g[0]][0], O[g[1]][0])
	x = grid(m)
	kk = np.arange(4 * n)
	cols = (BLUE, ORANGE)

	fig = plt.figure(figsize=(14.5, 8.2))
	gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.9], hspace=.3, wspace=.32)

	for c, (p, col) in enumerate(zip(pq, cols)):
		a = fig.add_subplot(gs[0, c])
		a.imshow(np.indices((n, n)).sum(0) % 2, cmap='Greys', alpha=.12)
		a.scatter(list(p), range(n), s=150, marker='s', color=col)
		a.set_xticks([]); a.set_yticks([])
		a.set_title(f'{"p" if c == 0 else "q"} = {list(p)}', color=col, fontsize=10)

	# el emparejamiento: mismos margenes, distinta nube
	a = fig.add_subplot(gs[1, :2])
	for p, col, mk in zip(pq, cols, ('o', 'x')):
		dp, dm = channels(p)
		a.scatter(dp, dm, s=130, marker=mk, color=col, lw=2,
				  facecolor='none' if mk == 'o' else col,
				  edgecolor=col, label=f'{"p" if mk == "o" else "q"}')
	dp0 = sorted(set(channels(pq[0])[0])); dm0 = sorted(set(channels(pq[0])[1]))
	for v in dp0:
		a.axvline(v, color=GREY, lw=.8, zorder=0)
	for v in dm0:
		a.axhline(v, color=GREY, lw=.8, zorder=0)
	a.set_xlabel('d+'); a.set_ylabel('d-'); a.legend(fontsize=9)
	a.set_title('las rectas grises son las diagonales ocupadas: IDENTICAS.\n'
				'lo que cambia es en que cruces se ponen las reinas', fontsize=10)

	a = fig.add_subplot(gs[0, 2])
	ca, cb = invariant(pq[0], x), invariant(pq[1], x)
	a.plot(x, ca[0], color=BLUE, lw=5, alpha=.4)
	a.plot(x, ca[1], color=BLUE, lw=5, alpha=.4)
	a.plot(x, cb[0], color=ORANGE, lw=1.4, ls='--')
	a.plot(x, cb[1], color=ORANGE, lw=1.4, ls='--')
	a.set_title(f'la curva invariante: identica (dif {np.abs(ca-cb).max():.0e})',
				fontsize=10)

	a = fig.add_subplot(gs[1, 2])
	sa = band_series(pq[0]) @ np.cos(np.outer(kk, x))
	sb = band_series(pq[1]) @ np.cos(np.outer(kk, x))
	a.plot(x, sa, color=BLUE, lw=5, alpha=.4)
	a.plot(x, sb, color=ORANGE, lw=1.4, ls='--')
	a.set_xlabel('x')
	a.set_title(f'y S(x) tambien: identica (dif {np.abs(sa-sb).max():.0e}).\n'
				'ninguna onda de una variable las separa', fontsize=10)

	fig.suptitle(f'N={n} — dos soluciones distintas, un solo espectro', y=.97)
	fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)
	print(f'-> {path}')


# ------------------------------------------------- forman las orbitas un patron?

def autocorr(S, L):
	v = np.zeros(L, dtype=int)
	S = sorted(S)
	for i in range(len(S)):
		for j in range(i + 1, len(S)):
			v[S[j] - S[i] - 1] += 1
	return v


def orbit_vectors(n):
	"""
	El invariante de una orbita, en su forma minima: el par de
	AUTOCORRELACIONES de los dos conjuntos de diagonales. |W|^2 es su
	transformada, asi que la curva no contiene nada mas que estos enteros.

	Cada canal: 2N-2 enteros >= 0 que suman N(N-1)/2.
	"""
	out = []
	for o in orbits(n):
		dp, dm = channels(o[0])
		a, b = autocorr(set(dp), 2 * n - 2), autocorr(set(dm), 2 * n - 2)
		out.append(np.concatenate(sorted([a, b], key=lambda v: tuple(v))))
	return np.array(out, float)


def dimension(ns=(8, 9, 10, 11)):
	"""
	Si las orbitas formasen una secuencia o un patron geometrico, viviria en
	dimension 1 o 2 y esa dimension no creceria con N. Medido: crece como ~2N.
	"""
	print(f'{"N":>3} {"orbitas":>8} {"coords":>7} {"rango":>6} {"dim 90%":>8} {"dim 99%":>8}')
	out = []
	for n in ns:
		V = orbit_vectors(n)
		C = V - V.mean(0)
		s = np.linalg.svd(C, compute_uv=False)
		cum = np.cumsum(s ** 2) / (s ** 2).sum()
		d90 = int(np.searchsorted(cum, .90)) + 1
		d99 = int(np.searchsorted(cum, .99)) + 1
		r = int(np.linalg.matrix_rank(C))
		print(f'{n:>3} {len(V):>8} {V.shape[1]:>7} {r:>6} {d90:>8} {d99:>8}')
		out.append((n, V, cum, r, d90))
	return out


def plot_dimension(ns=(8, 9, 10, 11), path='diag_waves/fig_orbit_dim.png'):
	plt = _mpl()
	rows = dimension(ns)
	fig, ax = plt.subplots(1, 2, figsize=(14, 5))
	cmap = plt.get_cmap('viridis')
	for j, (n, V, cum, r, d90) in enumerate(rows):
		col = cmap(j / len(rows))
		ax[0].plot(np.arange(1, len(cum) + 1), cum, 'o-', color=col, ms=3,
				   label=f'N={n}  (dim90 = {d90})')
	ax[0].axhline(.9, color='k', ls=':', lw=1)
	ax[0].set_xlim(0, 40); ax[0].set_xlabel('componentes principales')
	ax[0].set_ylabel('varianza acumulada'); ax[0].legend(fontsize=9)
	ax[0].set_title('un patron viviria en 1 o 2 dimensiones.\nla dimension crece con N')

	n, V = rows[1][0], rows[1][1]
	order = np.argsort(V @ np.arange(V.shape[1]))
	im = ax[1].imshow(V[order], aspect='auto', cmap='magma', interpolation='nearest')
	plt.colorbar(im, ax=ax[1], label='pares a esa distancia')
	ax[1].set_xlabel('coordenada del invariante (autocorrelacion, dos canales)')
	ax[1].set_ylabel(f'las {len(V)} orbitas de N={n}, ordenadas')
	ax[1].set_title('y ordenandolas como quieras, no aparece ninguna secuencia')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	check()
	print()
	plot_inside(8)
	plot_between(9)
	plot_collisions(9)
	plot_pair(9)
	plot_dimension()
