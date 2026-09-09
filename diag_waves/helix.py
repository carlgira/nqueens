"""
The helix: the complex wave Z(x) = sum_k c_k exp(i k x).

Everything we plotted so far was Re Z -- the shadow of this curve on one plane.
The full object is a curve in 3D: (x, Re Z, Im Z). Its mean squared radius is

	<|Z|^2> = sum_k c_k^2 = 2N + 2*conflicts

so solutions are the TIGHT helices (RMS radius sqrt(2N)) and every conflict
widens the tube. That is exactly the intuition: solutions coil close to the
axis, the rest swing wider.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import permutations

from diag_waves.diag_waves import band_series, channels, conflicts

C_GOOD, C_BAD = '#e8590c', '#adb5bd'


def helix(coef, x):
	k = np.arange(len(coef))
	return coef @ np.exp(1j * np.outer(k, x))


def plot(n=7, path='diag_waves/fig_helix.png', sample=500, seed=0):
	x = np.linspace(0, 2 * np.pi, 900)
	rng = np.random.default_rng(seed)
	sol, other, rs, ro = [], [], [], []
	for p in permutations(range(n)):
		dp, dm = channels(p)
		coef = band_series(p)
		z = helix(coef, x)
		# <|Z|^2> = sum c_k^2 exactamente (Parseval), sin sesgo de muestreo
		radius = np.sqrt(float((coef ** 2).sum()))
		if conflicts(dp) == 0 and conflicts(dm) == 0:
			sol.append(z); rs.append(radius)
		else:
			other.append(z); ro.append(radius)
	sol, other = np.array(sol), np.array(other)
	rms_s, rms_o = np.array(rs), np.array(ro)
	pick = other[rng.choice(len(other), sample, replace=False)]

	fig = plt.figure(figsize=(15.5, 12))

	ax = fig.add_subplot(2, 2, 1, projection='3d')
	ax.plot(x, sol[0].real, sol[0].imag, color=C_GOOD, lw=2, label='solución')
	ax.plot(x, other[0].real, other[0].imag, color='#495057', lw=1.4, label='no solución')
	ax.set_xlabel('x'); ax.set_ylabel('Re Z'); ax.set_zlabel('Im Z')
	ax.set_title('La hélice: $(x,\\ \\mathrm{Re}\\,Z,\\ \\mathrm{Im}\\,Z)$\n'
				 '$\\mathrm{Re}\\,Z = S(x)$, lo que veníamos graficando', fontsize=11)
	ax.legend(fontsize=9)

	ax2 = fig.add_subplot(2, 2, 2, projection='3d')
	for z in pick:
		ax2.plot(x, z.real, z.imag, color=C_BAD, lw=.35, alpha=.16)
	for z in sol:
		ax2.plot(x, z.real, z.imag, color=C_GOOD, lw=.9, alpha=.85)
	ax2.set_xlabel('x'); ax2.set_ylabel('Re Z'); ax2.set_zlabel('Im Z')
	ax2.set_title(f'{len(sol)} soluciones (naranja) dentro de {sample} permutaciones (gris)\n'
				  'las soluciones enrollan más apretado', fontsize=11)

	ax3 = fig.add_subplot(2, 2, 3)
	for z in pick:
		ax3.plot(z.real, z.imag, color=C_BAD, lw=.35, alpha=.16)
	for z in sol:
		ax3.plot(z.real, z.imag, color=C_GOOD, lw=.9, alpha=.85)
	th = np.linspace(0, 2 * np.pi, 200)
	r = np.sqrt(2 * n)
	ax3.plot(r * np.cos(th), r * np.sin(th), 'k--', lw=1.8,
			 label=f'radio RMS de toda solución = $\\sqrt{{2N}}$ = {r:.3f}')
	ax3.set_aspect('equal'); ax3.set_xlabel('Re Z'); ax3.set_ylabel('Im Z')
	ax3.set_title('Mirando por el eje: la roseta', fontsize=11)
	ax3.legend(fontsize=9, loc='upper right')

	ax4 = fig.add_subplot(2, 2, 4)
	bins = np.linspace(min(rms_o.min(), rms_s.min()) - .05, rms_o.max() + .05, 70)
	ax4.hist(rms_o, bins=bins, color=C_BAD, label=f'{len(other)} no soluciones')
	ax4.hist(rms_s, bins=bins, color=C_GOOD, label=f'{len(sol)} soluciones')
	ax4.axvline(r, color='k', ls='--', lw=1.6)
	ax4.set_yscale('log'); ax4.set_xlabel('radio RMS de la hélice')
	ax4.set_ylabel('nº de permutaciones (log)')
	ax4.set_title(f'radio RMS$^2$ = 2N + 2·conflictos\n'
				  f'soluciones: exactamente {r:.3f}   ·   resto: $\\geq$ {np.sqrt(2*n+2):.3f}',
				  fontsize=11)
	ax4.legend(fontsize=9)

	fig.tight_layout()
	fig.savefig(path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print(f'radio RMS soluciones: {rms_s.min():.6f} .. {rms_s.max():.6f}')
	print(f'radio RMS resto     : {rms_o.min():.6f} .. {rms_o.max():.6f}')
	return path


if __name__ == '__main__':
	print(plot(7))
