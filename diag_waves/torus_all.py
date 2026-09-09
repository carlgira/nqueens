"""Todas las permutaciones sobre el toro / la flor radial, en dos colores."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import permutations

from diag_waves.diag_waves import band_series, channels, conflicts

C_GOOD, C_BAD = '#e8590c', '#adb5bd'


def plot(n=7, pts=1500, path='diag_waves/fig_torus_all.png'):
	x = np.linspace(0, 2 * np.pi, pts)
	E = np.exp(1j * np.outer(np.arange(4 * n), x))
	sol, oth = [], []
	for p in permutations(range(n)):
		dp, dm = channels(p)
		z = band_series(p) @ E
		(sol if conflicts(dp) == 0 and conflicts(dm) == 0 else oth).append(z)
	sol, oth = np.array(sol), np.array(oth)
	Rs, Ro = np.abs(sol), np.abs(oth)

	fig = plt.figure(figsize=(17.5, 6.2))

	# --- flor radial, todas
	a = fig.add_subplot(1, 3, 1, projection='polar')
	a.add_collection(LineCollection([np.column_stack([x, r]) for r in Ro[::3]],
									colors=C_BAD, linewidths=.3, alpha=.06))
	for r in Rs:
		a.plot(x, r, color=C_GOOD, lw=.7, alpha=.75, zorder=3)
	a.set_ylim(0, 2 * n + .5)
	a.set_title(f'$|Z(x)|$ · las {len(sol)+len(oth)} permutaciones\n'
				f'naranja = {len(sol)} soluciones', fontsize=10.5, pad=16)
	a.set_yticklabels([])

	# --- dispersión: dónde se juntan las soluciones
	a2 = fig.add_subplot(1, 3, 2)
	sp_s = Rs.max(0) - Rs.min(0)
	sp_o = np.abs(np.vstack([sol, oth])).max(0) - np.abs(np.vstack([sol, oth])).min(0)
	a2.plot(x, sp_o, color=C_BAD, lw=1.2, label='todas las permutaciones')
	a2.plot(x, sp_s, color=C_GOOD, lw=1.6, label='solo soluciones')
	for m, lab in ((1, '0'), (3, '2π/3'), (2, 'π'), (1.5, '4π/3')):
		xv = 2 * np.pi / m if m != 1 else 0.0
		if m == 1.5: xv = 4 * np.pi / 3
		a2.axvline(xv, color='#1c7ed6', ls=':', lw=1.2)
		a2.text(xv + .05, sp_o.max() * .95, lab, color='#1c7ed6', fontsize=9)
	k = np.argsort(sp_s)[:400]
	a2.set_xlim(0, 2 * np.pi); a2.set_xlabel('x')
	a2.set_ylabel('dispersión de $|Z(x)|$')
	a2.set_title('Dónde se estrangulan\n'
				 f'mínimo de las soluciones: {sp_s.min():.3f} en x = {x[np.argmin(sp_s)]:.3f}',
				 fontsize=10.5)
	a2.legend(fontsize=9); a2.grid(alpha=.25)

	# --- el toro
	a3 = fig.add_subplot(1, 3, 3, projection='3d')
	u = np.linspace(0, 2 * np.pi, 80); v = np.linspace(0, 2 * np.pi, 36)
	U, V = np.meshgrid(u, v); r0, tube = 22., 7.
	a3.plot_surface((r0 + tube*np.cos(V))*np.cos(U), (r0 + tube*np.cos(V))*np.sin(U),
					tube*np.sin(V), color='#dee2e6', alpha=.15, linewidth=0, shade=False)
	def draw(Z, col, lw, al):
		rad = r0 + Z.real
		a3.plot(rad*np.cos(x), rad*np.sin(x), Z.imag, color=col, lw=lw, alpha=al)
	for z in oth[::60]: draw(z, C_BAD, .35, .10)
	for z in sol: draw(z, C_GOOD, .6, .55)
	a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([])
	a3.set_box_aspect((1, 1, .5)); a3.view_init(elev=34, azim=-58); a3.grid(False)
	for pane in (a3.xaxis, a3.yaxis, a3.zaxis): pane.pane.set_visible(False)
	a3.set_title('Sobre el toro', fontsize=10.5)

	fig.suptitle(f'N={n}: todas las permutaciones como curvas cerradas', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	print(f'dispersion minima de |Z| sobre soluciones: {sp_s.min():.4f} en x={x[np.argmin(sp_s)]:.4f}')
	print(f'  ({x[np.argmin(sp_s)]/np.pi:.4f} * pi)')
	return path


if __name__ == '__main__':
	print(plot())
