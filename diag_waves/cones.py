"""
Los conos: la region desde la que se cae en cada solucion.

Montaje bottom-up con las piezas que SI funcionan:
  1. potencial   = el testigo,  conflictos = (<S^2> - N)/2
  2. movimiento  = intercambio de dos columnas
  3. descenso    = al mejor vecino
  4. meseta      = paso lateral AL AZAR (el guiado determinista empeora: 54% vs 98%)
"""

import numpy as np
from itertools import combinations, permutations


def conflicts(p):
	n = len(p); i = np.arange(n)
	return (n - len(set((i + p).tolist()))) + (n - len(set((i - p).tolist())))


def descend(p, rng, side=40, cap=400):
	"""Devuelve (solucion o None, pasos)."""
	p = np.asarray(p).copy(); n = len(p); c = conflicts(p); used = 0
	for step in range(cap):
		if c == 0:
			return tuple(int(v) for v in p), step
		cand = []
		for i, j in combinations(range(n), 2):
			p[i], p[j] = p[j], p[i]; cc = conflicts(p); p[i], p[j] = p[j], p[i]
			cand.append((cc, i, j))
		m = min(x[0] for x in cand)
		if m > c:
			return None, step
		if m == c:
			used += 1
			if used > side:
				return None, step
		best = [x for x in cand if x[0] == m]
		_, i, j = best[rng.integers(len(best))]
		p[i], p[j] = p[j], p[i]; c = m
	return None, cap


def cone(sol, depth, rng, tries=400):
	"""Perfil del cono: a distancia k de `sol`, cuantos vuelven a `sol`."""
	n = len(sol); out = []
	for k in range(1, depth + 1):
		hit = 0
		for _ in range(tries):
			p = list(sol)
			for _ in range(k):
				i, j = rng.choice(n, 2, replace=False)
				p[i], p[j] = p[j], p[i]
			f, _s = descend(tuple(p), rng)
			hit += (f == tuple(sol))
		out.append(hit / tries)
	return out


def plot(path='diag_waves/fig_cones.png'):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	rng = np.random.default_rng(1)
	fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
	C1, C2, C3 = '#e8590c', '#1c7ed6', '#495057'

	# --- perfil del cono
	n = 11; sols, cur, full = [], [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if len(sols) >= 6: return
		if r == n: sols.append(tuple(cur)); return
		f = ~(cols | dp | dm) & full
		while f:
			b = f & -f; f ^= b; c = b.bit_length() - 1
			cur.append(c); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	ks = range(1, 6); S, O, F = [], [], []
	for k in ks:
		s = o = f_ = 0
		for sol in sols:
			for _ in range(120):
				p = list(sol)
				for _ in range(k):
					i, j = rng.choice(n, 2, replace=False); p[i], p[j] = p[j], p[i]
				r_, _st = descend(tuple(p), rng)
				if r_ is None: f_ += 1
				elif r_ == tuple(sol): s += 1
				else: o += 1
		t = s + o + f_; S.append(s/t); O.append(o/t); F.append(f_/t)
	a = ax[0]
	a.stackplot(list(ks), S, O, F, colors=[C1, C2, C3], alpha=.85,
				labels=['la MISMA solución', 'OTRA solución', 'atascado'])
	a.set_xlabel('pasos alejándose de una solución'); a.set_ylabel('fracción')
	a.set_xlim(1, 5); a.set_ylim(0, 1)
	a.set_title(f'N={n}: los conos TESELAN\nel gris (atascado) casi no existe', fontsize=11)
	a.legend(loc='center right', fontsize=9)

	# --- tamaños de cuenca
	from itertools import permutations as pp
	sinks = {}
	for p in pp(range(8)):
		r_, _ = descend(p, rng, side=0)
		if r_ is not None: sinks[r_] = sinks.get(r_, 0) + 1
	sz = np.array(sorted(sinks.values()))
	ax[1].bar(range(len(sz)), sz, color=C1, width=1.0)
	ax[1].axhline(sz.mean(), color='k', ls='--', lw=1.4, label=f'media {sz.mean():.0f}')
	ax[1].set_xlabel('cuenca (ordenadas por tamaño)'); ax[1].set_ylabel('permutaciones')
	ax[1].set_title(f'N=8: las {len(sz)} cuencas son casi iguales\n'
					f'min {sz.min()} · max {sz.max()} · razón {sz.max()/sz.min():.1f}', fontsize=11)
	ax[1].legend(fontsize=9)

	# --- rendimiento
	ns = [8, 12, 16, 20, 25]; suc, stp = [], []
	for m in ns:
		r_ = [descend(tuple(rng.permutation(m)), rng) for _ in range(150)]
		ok = [s for f_, s in r_ if f_ is not None]
		suc.append(len(ok)/len(r_)); stp.append(np.mean(ok))
	a3 = ax[2]
	a3.bar([str(v) for v in ns], suc, color=C1, alpha=.85, label='éxito')
	a3.set_ylim(0, 1.15); a3.set_ylabel('tasa de éxito'); a3.set_xlabel('N')
	a4 = a3.twinx(); a4.plot([str(v) for v in ns], stp, 'o-', color=C2, lw=2, label='pasos')
	a4.set_ylabel('pasos medios', color=C2); a4.set_ylim(0, max(stp)*1.6)
	a3.set_title('El solver montado\ndescenso + paso lateral al azar', fontsize=11)
	for i, v in enumerate(suc): a3.text(i, v+.03, f'{v:.0%}', ha='center', fontsize=9)

	fig.suptitle('Los conos: no buscar el punto, buscar la región que cae en él', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path
