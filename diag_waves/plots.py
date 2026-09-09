"""Figures for the diagonal-wave formulation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import permutations

from diag_waves.diag_waves import (channels, shift, occupancy, conflicts,
									band_series)

GRID = np.linspace(0, np.pi, 1500)			# S is even and 2pi-periodic
C_PLUS, C_MINUS, C_SUM = '#1c7ed6', '#0ca678', '#e8590c'
C_BAD, C_GOOD = '#adb5bd', '#e8590c'


def _series(coef, x):
	return coef @ np.cos(np.outer(np.arange(len(coef)), x))


def _channel_coefs(perm):
	n = len(perm)
	a, b = shift(*channels(perm), n)
	cp = np.zeros(4 * n)
	cm = np.zeros(4 * n)
	cp[:2 * n] = occupancy(a, n)
	cm[2 * n:] = occupancy(b, n)
	return cp, cm


def plot_two_and_sum(perms, labels, path='diag_waves/fig_two_and_sum.png'):
	"""Column per permutation; rows are R+, R- and their sum S."""
	n = len(perms[0])
	fig, axes = plt.subplots(3, len(perms), figsize=(6.4 * len(perms), 8.4),
							 sharex=True)
	axes = np.atleast_2d(axes.reshape(3, -1))

	for col, (perm, label) in enumerate(zip(perms, labels)):
		cp, cm = _channel_coefs(perm)
		dp, dm = channels(perm)
		k_p, k_m = conflicts(dp), conflicts(dm)
		rows = [
			(cp, C_PLUS,  f'$R^+(x)$   armónicos 1..{2*n-1}',       k_p, n / 2),
			(cm, C_MINUS, f'$R^-(x)$   armónicos {2*n+1}..{4*n-1}', k_m, n / 2),
			(cp + cm, C_SUM, '$S(x) = R^+(x) + R^-(x)$',   k_p + k_m, float(n)),
		]
		for row, (coef, color, title, k, floor) in enumerate(rows):
			ax = axes[row, col]
			ax.plot(GRID, _series(coef, GRID), color=color, lw=1.6)
			ax.axhline(0, color='#ced4da', lw=0.8, zorder=0)
			energy = 0.5 * float((coef ** 2).sum())		# exacto (Parseval, sin DC)
			head = f'{label}      perm = {list(perm)}\n\n' if row == 0 else ''
			ax.set_title(f'{head}{title}\n'
						 f'$\\langle\\cdot^2\\rangle$ = {energy:g} = {floor:g} + {k}'
						 f'    → conflictos = {k}', fontsize=10)
			ax.set_ylabel('amplitud', fontsize=9)
			ax.tick_params(labelsize=8)
			ax.margins(x=0)
		axes[2, col].set_xlabel('x   (0 a $\\pi$)', fontsize=9)

	fig.suptitle(f'N = {n}:  las dos ondas diagonales y su suma\n'
				 f'$\\langle R^\\pm{{}}^2\\rangle = N/2$ + conflictos de ese canal   ·   '
				 f'$\\langle S^2\\rangle = N$ + conflictos totales  (mínimo = {n})',
				 fontsize=13, y=0.99)
	fig.tight_layout(rect=(0, 0, 1, 0.90))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	return path


def plot_all(n=7, path='diag_waves/fig_all_n7.png'):
	"""Every permutation of N, drawn as its summed wave S(x)."""
	curves, energies, good = [], [], []
	basis = np.cos(np.outer(np.arange(4 * n), GRID))

	for p in permutations(range(n)):
		dp, dm = channels(p)
		k = conflicts(dp) + conflicts(dm)
		coef = band_series(p)
		curves.append(coef @ basis)
		energies.append(0.5 * float((coef ** 2).sum()))
		good.append(k == 0)

	curves = np.array(curves)
	energies = np.array(energies)
	good = np.array(good)

	fig = plt.figure(figsize=(15, 6.5))
	gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.18)
	ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

	seg_bad = [np.column_stack([GRID, y]) for y in curves[~good]]
	ax.add_collection(LineCollection(seg_bad, colors=C_BAD, linewidths=0.35,
									 alpha=0.10))
	for y in curves[good]:
		ax.plot(GRID, y, color=C_GOOD, lw=1.0, alpha=0.85, zorder=3)

	ax.set_xlim(0, np.pi)
	ax.set_ylim(curves.min() - 1, curves.max() + 1)
	ax.axhline(0, color='#495057', lw=0.7, zorder=2)
	ax.set_xlabel('x   (0 a $\\pi$)')
	ax.set_ylabel('$S(x)$')
	ax.set_title(f'Las {len(curves)} permutaciones de N={n}, cada una como su onda suma $S(x)$\n'
				 f'naranja = las {good.sum()} soluciones   ·   gris = las {(~good).sum()} restantes',
				 fontsize=12)
	ax.plot([], [], color=C_GOOD, lw=1.5, label='solución')
	ax.plot([], [], color='#868e96', lw=1.5, label='no solución')
	ax.legend(loc='upper right', fontsize=9)

	conf = np.rint(energies - n).astype(int)
	hi = conf.max()
	counts = np.bincount(conf, minlength=hi + 1)
	colors = [C_GOOD if i == 0 else C_BAD for i in range(hi + 1)]
	ax2.bar(np.arange(hi + 1) + n, counts, color=colors, width=0.85)
	ax2.set_yscale('log')
	ax2.set_xlabel('$\\langle S^2 \\rangle$  = N + conflictos totales')
	ax2.set_ylabel('nº de permutaciones (log)')
	ax2.set_title(f'La energía separa sin solapamiento\n'
				  f'soluciones = {n} exacto   ·   resto $\\geq$ {n+1}',
				  fontsize=12)
	ax2.axvline(n + 0.5, color='#495057', ls='--', lw=1.1)
	ax2.annotate(f'{counts[0]} soluciones', xy=(n, counts[0]),
				 xytext=(n + 3.2, counts[0] * 4.5), fontsize=10,
				 color=C_GOOD, fontweight='bold',
				 arrowprops=dict(arrowstyle='->', color=C_GOOD, lw=1.3))

	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	return path


def plot_bounds(n=7, path='diag_waves/fig_bounds.png'):
	"""Envelopes of S, and the bound computed from N alone via the moments."""
	from diag_waves.bounds import (moments, feasible_pairs, pair_wave,
									all_perm_waves, envelope)

	x = np.linspace(0, np.pi, 1500)
	m0, m1, m2 = moments(n)
	sol, other = all_perm_waves(n, x)
	lo_s, hi_s = envelope(sol)
	lo_a, hi_a = envelope(np.vstack([sol, other]))
	bound = np.array([pair_wave(p, n, x) for p in feasible_pairs(n)])
	lo_b, hi_b = envelope(bound)
	parab = m0 - m2 * x ** 2 / 2

	fig = plt.figure(figsize=(15, 9.5))
	gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.2)
	ax = fig.add_subplot(gs[0, :])

	ax.fill_between(x, lo_a, hi_a, color='#dee2e6', label='envolvente de las 5040 permutaciones')
	ax.fill_between(x, lo_s, hi_s, color=C_GOOD, alpha=0.85,
					label=f'envolvente real de las {len(sol)} soluciones')
	ax.plot(x, hi_b, color='#7048e8', lw=1.6, ls='--',
			label=f'COTA desde los momentos ({len(bound)} espectros factibles)')
	ax.plot(x, lo_b, color='#7048e8', lw=1.6, ls='--')
	ax.axhline(2 * n, color='#495057', ls=':', lw=1.2)
	ax.axhline(-2 * n, color='#495057', ls=':', lw=1.2)
	ax.text(np.pi * 0.985, 2 * n - 0.55, f'cota trivial $\\pm 2N = {2*n}$',
			ha='right', fontsize=9, color='#495057')
	t = x > .5
	top, bot = hi_b[t].max(), lo_b[t].min()
	ax.axhline(top, color='#c92a2a', ls='--', lw=1.3)
	ax.axhline(bot, color='#c92a2a', ls='--', lw=1.3)
	ax.text(np.pi * 0.985, top + 0.35, f'techo predicho {top:+.3f}',
			ha='right', fontsize=10, color='#c92a2a', fontweight='bold')
	ax.text(np.pi * 0.985, bot - 1.15, f'piso predicho {bot:+.3f}',
			ha='right', fontsize=10, color='#c92a2a', fontweight='bold')
	ax.set_xlim(0, np.pi)
	ax.set_ylim(-2 * n - 1.5, 2 * n + 1.5)
	ax.set_xlabel('x')
	ax.set_ylabel('$S(x)$')
	ax.set_title(f'N={n}: qué valores puede tomar $S(x)$\n'
				 f'la cota morada se calcula SOLO con N — sin resolver el problema',
				 fontsize=12)
	ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

	# --- zoom: the universal head and how it breaks
	az = fig.add_subplot(gs[1, 0])
	allw = np.vstack([sol, other])
	K4 = np.arange(4 * n) ** 4
	m4 = np.array([coef @ K4 for coef in _perm_iter(n, step=17)])
	zm = x <= 0.5
	for y in other[::37]:
		az.plot(x[zm], y[zm], color=C_BAD, lw=0.5, alpha=0.35)
	for y in sol:
		az.plot(x[zm], y[zm], color=C_GOOD, lw=0.9, alpha=0.8)
	t4 = m0 - m2 * x ** 2 / 2
	az.fill_between(x[zm],
					(t4 + m4.min() * x ** 4 / 24)[zm],
					(t4 + m4.max() * x ** 4 / 24)[zm],
					color='#4dabf7', alpha=0.35, zorder=4,
					label='Taylor 4º orden, $M_4$ de mín a máx')
	az.plot(x[zm], t4[zm], 'k--', lw=2.0, zorder=5,
			label=f'$M_0 - M_2x^2/2 = {m0} - {m2}x^2/2$   (universal)')
	spread = allw.max(axis=0) - allw.min(axis=0)
	xb = x[np.argmax(spread > 1.0)]
	az.axvline(xb, color='#1c7ed6', ls='-.', lw=1.4, zorder=6)
	az.text(xb + 0.012, 2 * n - 3.5, f'dispersión > 1\nen x = {xb:.2f}',
			fontsize=9, color='#1c7ed6', fontweight='bold')
	az.set_xlim(0, 0.5)
	az.set_ylim(-2 * n - 1, 2 * n + 1)
	az.set_xlabel('x')
	az.set_ylabel('$S(x)$')
	az.set_title('El arranque es universal ($M_0$, $M_2$ fijos);\n'
				 'la apertura la marca $M_4x^4/24$, el primer momento que varía',
				 fontsize=11)
	az.legend(loc='lower left', fontsize=8.5)

	# --- ceiling vs N
	ac = fig.add_subplot(gs[1, 1])
	ns = list(range(4, 11))
	pred, real = [], []
	for k in ns:
		xx = np.linspace(0, np.pi, 900)
		t = xx > 0.5
		bw = np.array([pair_wave(p, k, xx) for p in feasible_pairs(k)])
		sw = _solutions_waves(k, xx)
		pred.append(np.abs(bw[:, t]).max())
		real.append(np.abs(sw[:, t]).max())
	ac.plot(ns, [2 * k for k in ns], 'o:', color='#495057', label='cota trivial $2N$')
	ac.plot(ns, pred, 's--', color='#c92a2a', label='cota desde los momentos')
	ac.plot(ns, real, 'o-', color=C_GOOD, lw=2, label='máximo real de las soluciones')
	ac.set_xlabel('N')
	ac.set_ylabel('$\\max_{x>0.5} |S(x)|$')
	ac.set_title('La cota es exacta hasta N=9', fontsize=11)
	ac.legend(fontsize=9)
	ac.grid(alpha=0.25)

	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	return path


def plot_solutions_bound(n=9, path=None):
	"""Only the solutions of N, plus the moment bound, to judge the cover."""
	from diag_waves.bounds import feasible_pairs, pair_wave, envelope

	path = path or f'diag_waves/fig_bound_n{n}.png'
	x = np.linspace(0, np.pi, 2500)
	sw = _solutions_waves(n, x)
	lo_s, hi_s = envelope(sw)
	bw = np.array([pair_wave(p, n, x) for p in feasible_pairs(n)])
	lo_b, hi_b = envelope(bw)

	t = x > 0.5
	top, bot = hi_b[t].max(), lo_b[t].min()
	gap_hi, gap_lo = hi_b - hi_s, lo_s - lo_b

	fig = plt.figure(figsize=(15, 9))
	gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1], hspace=0.24)
	ax, ag = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

	ax.add_collection(LineCollection([np.column_stack([x, y]) for y in sw],
									 colors=C_GOOD, linewidths=0.6, alpha=0.22))
	ax.plot(x, hi_s, color='#d9480f', lw=1.1, label=f'envolvente de las {len(sw)} soluciones')
	ax.plot(x, lo_s, color='#d9480f', lw=1.1)
	ax.plot(x, hi_b, color='#7048e8', lw=1.8, ls='--',
			label=f'COTA desde los momentos ({len(bw)} espectros factibles)')
	ax.plot(x, lo_b, color='#7048e8', lw=1.8, ls='--')
	ax.axhline(top, color='#c92a2a', ls=':', lw=1.3)
	ax.axhline(bot, color='#c92a2a', ls=':', lw=1.3)
	ax.text(np.pi * .99, top + .3, f'techo {top:+.3f}', ha='right',
			fontsize=10, color='#c92a2a', fontweight='bold')
	ax.text(np.pi * .99, bot - .9, f'piso {bot:+.3f}', ha='right',
			fontsize=10, color='#c92a2a', fontweight='bold')
	ax.plot([], [], color=C_GOOD, lw=1.5, alpha=.5, label='las soluciones, una a una')
	ax.set_xlim(0, np.pi)
	ax.set_ylim(bot - 2, 2 * n + 1)
	ax.set_xlabel('x')
	ax.set_ylabel('$S(x)$')
	ax.set_title(f'N={n}: solo las {len(sw)} soluciones, con la cota calculada desde N',
				 fontsize=13)
	ax.legend(loc='upper right', fontsize=9.5, framealpha=.95)

	ag.fill_between(x, 0, gap_hi, color='#7048e8', alpha=.45, label='holgura arriba')
	ag.fill_between(x, 0, gap_lo, color='#1c7ed6', alpha=.45, label='holgura abajo')
	tight = (gap_hi < 1e-6) & (gap_lo < 1e-6)
	ag.set_xlim(0, np.pi)
	ag.set_xlabel('x')
	ag.set_ylabel('cota − real')
	ag.set_title(f'Holgura punto a punto  ·  la cota es exacta en el {tight.mean():.0%} del dominio  ·  '
				 f'peor caso {max(gap_hi.max(), gap_lo.max()):.2f}', fontsize=11)
	ag.legend(fontsize=9)
	ag.grid(alpha=.25)

	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	return path


def _perm_iter(n, step=1):
	"""Every `step`-th permutation, as band_series coefficient vectors."""
	return [band_series(p) for i, p in enumerate(permutations(range(n)))
			if i % step == 0]


def _solutions_waves(n, x):
	"""Waves of the solutions only (for N where enumerating N! is too slow)."""
	basis = np.cos(np.outer(np.arange(4 * n), x))
	out, col, dp, dm, cur = [], [False] * n, set(), set(), []

	def rec(r):
		if r == n:
			out.append(band_series(list(cur)) @ basis)
			return
		for c in range(n):
			if col[c] or (r + c) in dp or (r - c) in dm:
				continue
			col[c] = True; dp.add(r + c); dm.add(r - c); cur.append(c)
			rec(r + 1)
			col[c] = False; dp.discard(r + c); dm.discard(r - c); cur.pop()

	rec(0)
	return np.array(out)


if __name__ == '__main__':
	sol = [0, 2, 4, 6, 1, 3, 5]							# solución de N=7
	bad = [1, 3, 2, 6, 4, 5, 0]							# no lo es
	print(plot_two_and_sum([sol, bad], ['SOLUCIÓN', 'NO SOLUCIÓN']))
	print(plot_all(7))
	print(plot_bounds(7))
