"""Figuras de la representacion fraccional."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import frac
from cf import measures

BG = '#07070b'
SOL = '#ffb454'
PER = '#4a7fb5'
HERE = os.path.dirname(os.path.abspath(__file__))


def style(a, title=None):
	a.set_facecolor(BG)
	for s in a.spines.values():
		s.set_color('#3a3a4a')
	a.tick_params(colors='#8a8a9a', labelsize=8)
	a.xaxis.label.set_color('#b0b0c0'); a.yaxis.label.set_color('#b0b0c0')
	if title:
		a.set_title(title, color='#e6e6f0', fontsize=10, pad=8)


def xs(perms, n):
	return np.array([frac.frac(p, n) for p in perms])


def data(n, mult=1):
	"""Soluciones vs permutaciones que NO son solucion (mismo tamano de muestra)."""
	sols = frac.solutions(n)
	ss = set(sols)
	if n <= 8:
		perms = [p for p in frac.permutations_all(n) if p not in ss]
	else:
		perms = [p for p in frac.permutations_sample(n, mult * len(sols) + 50, seed=1)
		         if p not in ss][:mult * len(sols)]
	return sols, perms


# ------------------------------------------------------------------ fig 1
def fig_curve():
	fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=BG)
	for j, n in enumerate((8, 10, 12)):
		sols, perms = data(n)
		s, p = np.sort(xs(sols, n)), np.sort(xs(perms, n))
		a = axes[0, j]; style(a, f"N={n}: los {len(s)} numeros ordenados")
		a.plot(np.linspace(0, 1, len(p)), p, color=PER, lw=1.2, label='permutaciones')
		a.plot(np.linspace(0, 1, len(s)), s, color=SOL, lw=1.4, label='soluciones')
		a.plot([0, 1], [0, 1], color='#555566', lw=0.7, ls='--')
		a.set_xlabel('rango / total'); a.set_ylabel('x')
		a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')

		a = axes[1, j]; style(a, f"densidad de x  ({n} bandas del primer digito)")
		bins = np.linspace(0, 1, n * n + 1)
		a.hist(p, bins=bins, density=True, color=PER, alpha=.55, label='permutaciones')
		a.hist(s, bins=bins, density=True, color=SOL, alpha=.8,
		       histtype='stepfilled', label='soluciones')
		for k in range(1, n):
			a.axvline(k / n, color='#33334a', lw=.5)
		a.set_xlabel('x'); a.set_ylabel('densidad')
		a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')
	fig.suptitle('x(sigma) = 0.sigma(0)sigma(1)...sigma(N-1) en base N   '
	             '-- la curva de las soluciones se aplana en los extremos',
	             color='#e6e6f0', fontsize=12)
	fig.tight_layout(rect=[0, 0, 1, .95])
	fig.savefig(f'{HERE}/fig_curve.png', dpi=130, facecolor=BG)
	print('fig_curve.png')


# ------------------------------------------------------------------ fig 2
def fig_plane():
	fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=BG)
	n = 13
	sols = frac.solutions(n)
	perms = frac.permutations_sample(n, len(sols), seed=2)

	def inv(s):
		o = [0] * len(s)
		for k, v in enumerate(s):
			o[v] = k
		return tuple(o)

	for a, ps, c, t in ((axes[0], perms, PER, f'N={n}: {len(perms)} permutaciones al azar'),
	                    (axes[1], sols, SOL, f'N={n}: las {len(sols)} soluciones')):
		style(a, t)
		X = xs(ps, n); Y = xs([inv(s) for s in ps], n)
		a.scatter(X, Y, s=.6, c=c, alpha=.5, linewidths=0)
		a.set_xlabel('x(sigma)'); a.set_ylabel('x(sigma^-1)  [tablero transpuesto]')
		a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect('equal')
	a = axes[2]; style(a, 'zoom de las soluciones: la banda prohibida es fractal')
	X = xs(sols, n); Y = xs([inv(s) for s in sols], n)
	m = (X > .45) & (X < .55) & (Y > .45) & (Y < .55)
	a.scatter(X[m], Y[m], s=4, c=SOL, alpha=.7, linewidths=0)
	a.set_xlabel('x(sigma)'); a.set_ylabel('x(sigma^-1)'); a.set_aspect('equal')
	fig.tight_layout()
	fig.savefig(f'{HERE}/fig_plane.png', dpi=130, facecolor=BG)
	print('fig_plane.png')


# ------------------------------------------------------------------ fig 3
def fig_cf():
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=BG)
	n = 12
	sols, perms = data(n)
	S = np.array([measures(s, n) for s in sols], float)
	P = np.array([measures(s, n) for s in perms], float)

	a = axes[0]; style(a, f'N={n}: plano de la fraccion continua')
	a.scatter(P[:, 0], P[:, 1], s=2, c=PER, alpha=.35, linewidths=0, label='permutaciones')
	a.scatter(S[:, 0], S[:, 1], s=2, c=SOL, alpha=.45, linewidths=0, label='soluciones')
	for i in np.argsort(-S[:, 1])[:2]:
		a.annotate(f"paso {sorted({(sols[i][k+1]-sols[i][k])%n for k in range(n-1)})}",
		           (S[i, 0], S[i, 1]), color='#ff6f6f', fontsize=8,
		           xytext=(S[i, 0] + .15, S[i, 1]), arrowprops=dict(color='#ff6f6f', lw=.7,
		           arrowstyle='->'))
	a.set_xlabel('L = longitud FC / 0.8427 ln q     (1 = generico)')
	a.set_ylabel('C = ln(max a_i)/ln q     (0 = generico)')
	a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')

	a = axes[1]; style(a, 'cola: P(C > c)')
	for D, c, lab in ((P, PER, 'permutaciones'), (S, SOL, 'soluciones')):
		v = np.sort(D[:, 1])
		a.semilogy(v, 1 - np.arange(len(v)) / len(v), color=c, lw=1.4, label=lab)
	a.set_xlabel('c'); a.set_ylabel('fraccion con C > c')
	a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')

	a = axes[2]; style(a, 'media geometrica de los cocientes parciales')
	bins = np.linspace(1.5, 6, 60)
	a.hist(P[:, 2], bins=bins, density=True, color=PER, alpha=.6, label='permutaciones')
	a.hist(S[:, 2], bins=bins, density=True, color=SOL, alpha=.6, label='soluciones')
	a.axvline(2.6854, color='#ff6f6f', lw=1, ls='--')
	a.annotate('Khinchin 2.685', (2.72, a.get_ylim()[1] * .8), color='#ff6f6f', fontsize=8)
	a.set_xlabel('G'); a.set_ylabel('densidad')
	a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')
	fig.tight_layout()
	fig.savefig(f'{HERE}/fig_cf.png', dpi=130, facecolor=BG)
	print('fig_cf.png')


# ------------------------------------------------------------------ fig 4
def fig_gaps_dim():
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=BG)
	a = axes[0]; style(a, 'huecos entre numeros consecutivos (normalizados)')
	for n, c in ((10, '#7fd4ff'), (12, SOL), (13, '#ff8fb0')):
		sols, perms = data(n)
		for ps, ls, lab in ((sols, '-', 'sol'), (perms, ':', 'perm')):
			v = np.sort(xs(ps, n)); g = np.diff(v) * len(v)
			g = np.sort(g[g > 0])
			a.loglog(g, 1 - np.arange(len(g)) / len(g), color=c, lw=1.2, ls=ls,
			         label=f'N={n} {lab}')
	t = np.logspace(-3, 1, 50)
	a.loglog(t, np.exp(-t), color='#ffffff', lw=.8, ls='--', label='e^-t (Poisson)')
	a.set_xlabel('hueco / hueco medio'); a.set_ylabel('P(hueco > t)')
	a.set_ylim(1e-4, 1.2)
	a.legend(fontsize=6, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0', ncol=2)

	a = axes[1]; style(a, 'dimension de caja: cajas de lado N^-d')
	for n in range(7, 14):
		sols = frac.solutions(n)
		dd, vv = [], []
		for d in range(1, n + 1):
			B = len({s[:d] for s in sols})
			dd.append(d / n); vv.append(math.log(B) / (d * math.log(n)))
		a.plot(dd, vv, marker='o', ms=2.5, lw=1, label=f'N={n}')
	a.set_xlabel('d / N'); a.set_ylabel('log B_d / (d log N)')
	a.legend(fontsize=6, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0', ncol=2)

	a = axes[2]; style(a, 'dimension final vs 1 - 1.944/ln N  (Simkin)')
	ns = np.arange(7, 14)
	dims = []
	for n in ns:
		q = len(frac.solutions(n))
		dims.append(math.log(q) / (n * math.log(n)))
	a.plot(ns, dims, marker='o', color=SOL, label='log Q(N) / (N log N)')
	a.plot(ns, 1 - 1.944 / np.log(ns), color='#ff6f6f', ls='--',
	       label='asintota 1 - alpha/ln N')
	a.set_xlabel('N'); a.set_ylabel('dimension')
	a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')
	fig.tight_layout()
	fig.savefig(f'{HERE}/fig_gaps_dim.png', dpi=130, facecolor=BG)
	print('fig_gaps_dim.png')


def fig_scaling():
	import scaling
	f = f'{HERE}/scaling.npy'
	R = np.load(f) if os.path.exists(f) else scaling.run()
	fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=BG)
	a = axes[0]
	style(a, 'C = ln(max a_i)/ln q  segun N   (0 = numero generico)')
	a.plot(R[:, 0], R[:, 3], color='#ff6f6f', lw=1.6, label='mejor progresion aritmetica')
	a.plot(R[:, 0], R[:, 2], color=SOL, lw=1.3, label='construccion clasica, transpuesta')
	a.plot(R[:, 0], R[:, 1], color='#c98a2e', lw=1.3, label='construccion clasica')
	a.plot(R[:, 0], R[:, 4], color='#7fd4ff', lw=1.3, label='solucion generica')
	a.plot(R[:, 0], R[:, 5], color=PER, lw=1.3, ls=':', label='permutacion al azar')
	a.set_xlabel('N'); a.set_ylabel('C'); a.set_ylim(0, 1)
	a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')

	a = axes[1]
	style(a, 'longitud de la fraccion continua')
	ns = R[:, 0]
	import math as _m
	from cf import HEILBRONN
	a.plot(ns, [len(frac.cont_frac(*frac.reduced(scaling.classic(int(n)), int(n))))
	            for n in ns], color=SOL, lw=1.3, label='construccion clasica')
	a.plot(ns, [len(frac.cont_frac(*frac.reduced(
	            scaling.inverse(scaling.classic(int(n))), int(n)))) for n in ns],
	       color='#c98a2e', lw=1.3, label='su transpuesta')
	rng = __import__('random').Random(11)
	a.plot(ns, [len(frac.cont_frac(*frac.reduced(
	            tuple(rng.sample(range(int(n)), int(n))), int(n)))) for n in ns],
	       color=PER, lw=1.3, label='permutacion al azar')
	a.plot(ns, [HEILBRONN * int(n) * _m.log(int(n)) for n in ns], color='#ff6f6f',
	       ls='--', lw=1, label='0.8427 N ln N  (Heilbronn)')
	a.set_xlabel('N'); a.set_ylabel('numero de cocientes parciales')
	a.legend(fontsize=7, facecolor=BG, edgecolor='#3a3a4a', labelcolor='#c0c0d0')
	fig.tight_layout()
	fig.savefig(f'{HERE}/fig_scaling.png', dpi=130, facecolor=BG)
	print('fig_scaling.png')


if __name__ == '__main__':
	fig_curve(); fig_cf(); fig_gaps_dim(); fig_plane(); fig_scaling()
