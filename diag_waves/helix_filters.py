"""Directional radii, lattice bridges and density of the helix Z(x)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import permutations

from diag_waves.diag_waves import band_series, channels, conflicts

C_GOOD, C_BAD = '#e8590c', '#adb5bd'


def curves(n, pts=4000):
	x = np.linspace(0, 2 * np.pi, pts, endpoint=False)
	E = np.exp(1j * np.outer(np.arange(4 * n), x))
	sol, oth = [], []
	for p in permutations(range(n)):
		dp, dm = channels(p)
		(sol if conflicts(dp) == 0 and conflicts(dm) == 0 else oth).append(band_series(p) @ E)
	return x, np.array(sol), np.array(oth)


def plot(n=7, path='diag_waves/fig_helix_filters.png'):
	x, sol, oth = curves(n)
	body = (x > 0.5) & (x < 2 * np.pi - 0.5)		# fuera del arranque universal
	S, O = sol[:, body], oth[:, body]

	fig, ax = plt.subplots(2, 2, figsize=(15, 13))

	# --- roseta con radios direccionales
	a = ax[0, 0]
	for z in oth[::13]:
		a.plot(z.real, z.imag, color=C_BAD, lw=.3, alpha=.10)
	for z in sol:
		a.plot(z.real, z.imag, color=C_GOOD, lw=.8, alpha=.8)
	lim = {'Re max': (S.real.max(), O.real.max()), 'Re min': (S.real.min(), O.real.min()),
		   'Im max': (S.imag.max(), O.imag.max()), 'Im min': (S.imag.min(), O.imag.min())}
	for k, (vs, vo) in lim.items():
		f = a.axvline if k.startswith('Re') else a.axhline
		f(vs, color=C_GOOD, ls='--', lw=1.6)
		f(vo, color='#495057', ls=':', lw=1.3)
	a.set_aspect('equal'); a.set_xlabel('Re Z'); a.set_ylabel('Im Z')
	a.set_title('Radios por dirección (fuera del arranque)\n'
				f'naranja = soluciones, punteado gris = todas', fontsize=11)
	txt = '\n'.join(f'{k}:  sol {vs:+7.3f}   resto {vo:+7.3f}' for k, (vs, vo) in lim.items())
	a.text(.02, .02, txt, transform=a.transAxes, fontsize=8.5, family='monospace',
		   va='bottom', bbox=dict(fc='white', alpha=.85, ec='#ced4da'))

	# --- puentes: puntos racionales
	a = ax[0, 1]
	for z in oth[::13]:
		a.plot(z.real, z.imag, color=C_BAD, lw=.3, alpha=.08)
	for z in sol:
		a.plot(z.real, z.imag, color=C_GOOD, lw=.5, alpha=.35)
	K = np.arange(4 * n)
	cols = {3: '#1c7ed6', 4: '#0ca678', 6: '#7048e8'}
	for m, c in cols.items():
		e = np.exp(1j * K * 2 * np.pi / m)
		pts_s = {complex(np.round(band_series(p) @ e, 6)) for p in permutations(range(n))
				 if conflicts(channels(p)[0]) == 0 and conflicts(channels(p)[1]) == 0}
		pts_a = {complex(np.round(band_series(p) @ e, 6)) for p in permutations(range(n))}
		a.plot([v.real for v in pts_a], [v.imag for v in pts_a], 'x', color=c, ms=7, mew=1.4,
			   alpha=.5)
		a.plot([v.real for v in pts_s], [v.imag for v in pts_s], 'o', color=c, ms=10,
			   mec='k', mew=.8, label=f'x=2$\\pi$/{m}: {len(pts_s)} de {len(pts_a)}')
	a.set_aspect('equal'); a.set_xlabel('Re Z'); a.set_ylabel('Im Z')
	a.set_title('Los puentes: en $x=2\\pi/m$ la hélice cae en una retícula\n'
				'círculo = alcanzado por soluciones · aspa = solo por otras', fontsize=11)
	a.legend(fontsize=9, loc='upper left')

	# --- densidad
	a = ax[1, 0]
	h, xe, ye = np.histogram2d(S.real.ravel(), S.imag.ravel(), bins=180)
	a.imshow(np.log1p(h.T), origin='lower', cmap='magma',
			 extent=(xe[0], xe[-1], ye[0], ye[-1]), aspect='equal')
	a.set_xlabel('Re Z'); a.set_ylabel('Im Z')
	a.set_title('Dónde se concentran las soluciones (log-densidad)', fontsize=11)

	# --- radio instantaneo
	a = ax[1, 1]
	a.fill_between(x[body], np.abs(O).min(axis=0), np.abs(O).max(axis=0),
				   color=C_BAD, label='|Z| de todas las permutaciones')
	a.fill_between(x[body], np.abs(S).min(axis=0), np.abs(S).max(axis=0),
				   color=C_GOOD, alpha=.85, label='|Z| de las soluciones')
	a.axhline(np.sqrt(2 * n), color='k', ls='--', lw=1.6,
			  label=f'radio RMS $\\sqrt{{2N}}$ = {np.sqrt(2*n):.3f}')
	a.set_xlabel('x'); a.set_ylabel('$|Z(x)|$')
	a.set_title('El radio instantáneo del tubo', fontsize=11)
	a.legend(fontsize=9)

	fig.suptitle(f'N={n}: filtros sobre la hélice $Z(x)=\\sum_k c_k e^{{ikx}}$', fontsize=14)
	fig.tight_layout(rect=(0, 0, 1, .96))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	for k, (vs, vo) in lim.items():
		print(f'  {k}: soluciones {vs:+8.3f}   todas {vo:+8.3f}')
	return path


if __name__ == '__main__':
	print(plot(7))
