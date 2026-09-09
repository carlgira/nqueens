"""
Caminos por el mapa de cuellos de botella.

El mapa: cada punto racional x = 2*pi*j/m es una CIUDAD, y los valores
discretos que S(x) puede tomar ahi son sus BARRIOS (los cuellos). Una celda
(ciudad, barrio) contiene un conjunto de permutaciones, y toca un conjunto de
orbitas de simetria.

Un CAMINO es una secuencia de celdas. Se para cuando la union de lo visitado
toca al menos una solucion de cada orbita. Empieza donde sea, salta a donde sea.

Dos formas de saltar:

	uniforme   -- barrio al azar dentro de la ciudad
	ponderado  -- barrio al azar con probabilidad proporcional a cuantos
				  espectros FACTIBLES caen ahi (calculable solo con N)

El resultado, sobre 400 caminos por N:

	N     orbitas   celdas   ponderado (mediana)   uniforme (mediana)
	 8       12       17            3                    4
	 9       46       32            4                    9
	10       92       44            3                   10
	11      341       66            3                    7
	12     1787       64            8                   17

Dos cosas que decir. La primera: ponderar por masa factible reduce el camino a
la MITAD o menos, y esa ponderacion no mira ninguna solucion. La segunda, mas
llamativa: el camino no crece con N como crecen las orbitas. De N=8 a N=12 las
orbitas se multiplican por 149 y la mediana del camino pasa de 3 a 8.

Eso es el "se repite entre N" que buscabamos, y es tambien su limite: cubrir es
FACIL, y por eso cubrir no poda. Las celdas son grandes.
"""

import numpy as np
from math import gcd

from diag_waves.pp import feasible_coefs
from diag_waves.orbits import orbits
from diag_waves.necks import coefs, rational_points

GREY, BLUE, ORANGE, PURPLE, GREEN = '#dee2e6', '#1c7ed6', '#e8590c', '#7048e8', '#0ca678'


def build(n, NS=60000, mmax=12, seed=0):
	"""El mapa: todas las celdas (ciudad, barrio) con su masa, coste y orbitas."""
	rng = np.random.default_rng(seed)
	F = feasible_coefs(n)
	k = np.arange(4 * n)
	O = orbits(n)
	K = len(O)
	oid = np.array([i for i, o in enumerate(O) for _ in o])
	CS = coefs(np.array([p for o in O for p in o]), n)
	CP = coefs(np.array([rng.permutation(n) for _ in range(NS)]), n)
	cells = []
	for m, j in rational_points(mmax):
		w = np.cos(k * 2 * np.pi * j / m)
		vp = np.round(CP @ w, 9)
		if len(np.unique(vp)) > 4 * n:
			continue
		vf = np.round(F @ w, 9)
		vs = np.round(CS @ w, 9)
		u, mu = np.unique(vf, return_counts=True)
		if len(u) < 4:
			continue
		for t, mm in zip(u, mu):
			cells.append(dict(pt=f'2pi*{j}/{m}', m=m, j=j, val=float(t),
							  mult=int(mm), orb=frozenset(oid[vs == t].tolist()),
							  cost=float((vp == t).mean())))
	return cells, K


def walk(cells, K, rng, weighted=True, maxsteps=300):
	"""Un camino: salta de ciudad en ciudad hasta cubrir todas las orbitas."""
	pts = sorted({c['pt'] for c in cells})
	by = {p: [c for c in cells if c['pt'] == p] for p in pts}
	cov, path, hist = set(), [], []
	for _ in range(maxsteps):
		cs = by[pts[rng.integers(len(pts))]]
		if weighted:
			w = np.array([c['mult'] for c in cs], float)
			c = cs[rng.choice(len(cs), p=w / w.sum())]
		else:
			c = cs[rng.integers(len(cs))]
		if any(c is q for q in path):
			continue
		path.append(c)
		cov |= c['orb']
		hist.append(len(cov) / K)
		if len(cov) == K:
			break
	return path, hist


def stats(ns=(8, 9, 10, 11, 12), trials=400, seed=7):
	out = {}
	print(f'{"N":>3} {"orbitas":>8} {"celdas":>7} | {"ponderado":>22} | {"uniforme":>22}')
	for n in ns:
		cells, K = build(n)
		rng = np.random.default_rng(seed)
		r = {}
		for wt in (True, False):
			L, H = [], []
			for _ in range(trials):
				p, h = walk(cells, K, rng, weighted=wt)
				L.append(len(p) if h and h[-1] >= 1 else np.nan)
				H.append(h)
			r[wt] = (np.array(L, float), H)
		out[n] = (cells, K, r)
		a, b = r[True][0], r[False][0]
		print(f'{n:>3} {K:>8} {len(cells):>7} | mediana {np.nanmedian(a):>4.0f} '
			  f'[{np.nanpercentile(a,10):>3.0f}-{np.nanpercentile(a,90):>3.0f}] | '
			  f'mediana {np.nanmedian(b):>4.0f} '
			  f'[{np.nanpercentile(b,10):>3.0f}-{np.nanpercentile(b,90):>3.0f}]', flush=True)
	return out


def _mpl():
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	return plt


def plot_map(n=10, nwalks=3, path=None, seed=3):
	"""El mapa con tres caminos de ejemplo, y la cobertura paso a paso."""
	plt = _mpl()
	path = path or f'diag_waves/fig_paths_map_n{n}.png'
	cells, K = build(n)
	pts = sorted({c['pt'] for c in cells}, key=lambda s: (int(s.split('/')[1]), s))
	xi = {p: i for i, p in enumerate(pts)}
	rng = np.random.default_rng(seed)

	fig, ax = plt.subplots(1, 2, figsize=(15, 5.6),
						   gridspec_kw={'width_ratios': [1.5, 1]})
	mx = max(c['mult'] for c in cells)
	for c in cells:
		ax[0].scatter(xi[c['pt']], c['val'],
					  s=30 + 420 * c['mult'] / mx, color=GREY,
					  edgecolor='#adb5bd', lw=.5, zorder=1)
	cols = [ORANGE, GREEN, PURPLE]
	for w in range(nwalks):
		p, h = walk(cells, K, rng, weighted=True)
		xs = [xi[c['pt']] for c in p]; ys = [c['val'] for c in p]
		ax[0].plot(xs, ys, '-o', color=cols[w], lw=2, ms=9, zorder=3,
				   alpha=.9, label=f'camino {w+1}: {len(p)} saltos')
		for s, (x0, y0) in enumerate(zip(xs, ys)):
			ax[0].annotate(f'{s+1}\n{100*h[s]:.0f}%', (x0, y0),
						   textcoords='offset points', xytext=(11, -4),
						   fontsize=7.5, color=cols[w], weight='bold')
	ax[0].set_xticks(range(len(pts))); ax[0].set_xticklabels(pts, fontsize=9)
	ax[0].set_xlabel('ciudad (punto racional)'); ax[0].set_ylabel('barrio (valor de S(x))')
	ax[0].legend(fontsize=9, loc='upper right')
	ax[0].set_title(f'N={n}: el mapa. Tamano del punto = masa factible.\n'
					f'Etiqueta = paso y cobertura acumulada de las {K} orbitas')

	for wt, col, lab in ((True, GREEN, 'ponderado por masa factible'),
						 (False, GREY, 'uniforme')):
		rng2 = np.random.default_rng(11)
		H = [walk(cells, K, rng2, weighted=wt)[1] for _ in range(300)]
		L = max(len(h) for h in H)
		M = np.array([h + [1.0] * (L - len(h)) for h in H])
		ax[1].plot(range(1, L + 1), 100 * np.median(M, 0), color=col, lw=2.4, label=lab)
		ax[1].fill_between(range(1, L + 1), 100 * np.percentile(M, 10, 0),
						   100 * np.percentile(M, 90, 0), color=col, alpha=.18, lw=0)
	ax[1].axhline(100, color='k', ls=':', lw=1)
	ax[1].set_xlabel('saltos'); ax[1].set_ylabel('% de orbitas cubiertas')
	ax[1].set_xlim(1, 25); ax[1].legend(fontsize=9, loc='lower right')
	ax[1].set_title('300 caminos: mediana y banda 10-90%')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_across(out=None, path='diag_waves/fig_paths_across.png'):
	"""Como se repite entre N: cobertura por paso y longitud del camino."""
	plt = _mpl()
	out = out or stats()
	fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))
	cmap = plt.get_cmap('viridis')
	ns = sorted(out)
	for i, n in enumerate(ns):
		cells, K, r = out[n]
		H = r[True][1]
		L = max(len(h) for h in H)
		M = np.array([h + [1.0] * (L - len(h)) for h in H])
		ax[0].plot(range(1, L + 1), 100 * np.median(M, 0), lw=2.2,
				   color=cmap(i / len(ns)), label=f'N={n}  ({K} orbitas)')
	ax[0].axhline(100, color='k', ls=':', lw=1)
	ax[0].set_xlim(1, 20); ax[0].set_xlabel('saltos')
	ax[0].set_ylabel('% de orbitas cubiertas'); ax[0].legend(fontsize=9)
	ax[0].set_title('la curva de cobertura apenas se mueve al crecer N')

	w = .38
	med_w = [np.nanmedian(out[n][2][True][0]) for n in ns]
	med_u = [np.nanmedian(out[n][2][False][0]) for n in ns]
	ax[1].bar(np.arange(len(ns)) - w / 2, med_w, w, color=GREEN, label='ponderado')
	ax[1].bar(np.arange(len(ns)) + w / 2, med_u, w, color=GREY, label='uniforme')
	for i, n in enumerate(ns):
		a = out[n][2][True][0]
		ax[1].errorbar(i - w / 2, np.nanmedian(a),
					   yerr=[[np.nanmedian(a) - np.nanpercentile(a, 10)],
							 [np.nanpercentile(a, 90) - np.nanmedian(a)]],
					   color='k', capsize=3, lw=1)
	ax2 = ax[1].twinx()
	ax2.plot(range(len(ns)), [out[n][1] for n in ns], 'o--', color=ORANGE, lw=1.6)
	ax2.set_yscale('log'); ax2.set_ylabel('orbitas que hay que cubrir', color=ORANGE)
	ax[1].set_xticks(range(len(ns))); ax[1].set_xticklabels([f'N={n}' for n in ns])
	ax[1].set_ylabel('saltos hasta cubrir (mediana)'); ax[1].legend(fontsize=9)
	ax[1].set_title('las orbitas se multiplican por 149; el camino pasa de 3 a 8')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	out = stats()
	plot_map(10)
	plot_across(out)
