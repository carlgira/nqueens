"""
La hélice no es una hélice: Z(x) es 2*pi-periodica, asi que Z(0) = Z(2*pi) y la
curva se cierra. El eje x es un CIRCULO, no una recta. La representacion honesta
es una curva cerrada sobre un toro -- y su sombra radial es otra flor.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import band_series, channels, conflicts

C_GOOD, C_BAD = '#e8590c', '#495057'


def curve(perm, pts=4000):
	c = band_series(perm)
	x = np.linspace(0, 2 * np.pi, pts)
	return x, c @ np.exp(1j * np.outer(np.arange(len(c)), x))


def torus(x, Z, R=22.0, s=1.0):
	rad = R + s * Z.real
	return rad * np.cos(x), rad * np.sin(x), s * Z.imag


def plot(n=9, good=(0, 2, 5, 7, 1, 3, 8, 6, 4), bad=(0, 1, 3, 6, 2, 8, 4, 7, 5),
		 path='diag_waves/fig_torus.png'):
	fig = plt.figure(figsize=(17, 5.8))

	# --- 1. la flor radial |Z(x)|
	a = fig.add_subplot(1, 3, 1, projection='polar')
	for perm, col, lab in ((list(good), C_GOOD, 'solución'), (list(bad), C_BAD, 'no solución')):
		x, Z = curve(perm)
		dp, dm = channels(perm)
		k = conflicts(dp) + conflicts(dm)
		a.plot(x, np.abs(Z), color=col, lw=1.4, label=f'{lab}  ({k} conflictos)')
	a.set_title('$|Z(x)|$ en polar: la flor radial\n'
				'el eje x enrollado en un círculo', fontsize=10.5, pad=16)
	a.legend(fontsize=8.5, loc='lower right', bbox_to_anchor=(1.2, -.12))

	# --- 2 y 3. la curva sobre el toro
	for j, (perm, col, lab) in enumerate(((list(good), C_GOOD, 'SOLUCIÓN'),
										  (list(bad), C_BAD, 'NO SOLUCIÓN'))):
		ax = fig.add_subplot(1, 3, 2 + j, projection='3d')
		u = np.linspace(0, 2 * np.pi, 90); v = np.linspace(0, 2 * np.pi, 40)
		U, V = np.meshgrid(u, v)
		r0, tube = 22.0, 7.0
		ax.plot_surface((r0 + tube * np.cos(V)) * np.cos(U),
						(r0 + tube * np.cos(V)) * np.sin(U),
						tube * np.sin(V), color='#dee2e6', alpha=.18,
						linewidth=0, shade=False)
		x, Z = curve(perm)
		X, Y, W = torus(x, Z)
		ax.plot(X, Y, W, color=col, lw=1.1)
		dp, dm = channels(perm)
		k = conflicts(dp) + conflicts(dm)
		ax.set_title(f'{lab}  {perm}\n{k} conflictos   ·   '
					 f'$\\langle|Z|^2\\rangle$ = {2*n + 2*k}', fontsize=10, color=col)
		ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
		ax.set_box_aspect((1, 1, .55)); ax.view_init(elev=32, azim=-58)
		ax.grid(False)
		for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
			pane.pane.set_visible(False)

	fig.suptitle(f'N={n}: la curva cerrada sobre el toro  ·  '
				 '$Z(0)=Z(2\\pi)$, el eje $x$ es un círculo', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
