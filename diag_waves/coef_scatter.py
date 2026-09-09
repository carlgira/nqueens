"""Los coeficientes de Fourier de la hélice de una solución, en 2D."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import band_series


def first_sol(n):
	full, cur = (1 << n) - 1, []
	def rec(r, cols, dp, dm):
		if r == n: return True
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b; c = b.bit_length() - 1
			cur.append(c)
			if rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1): return True
			cur.pop()
		return False
	rec(0, 0, 0, 0); return cur


def plot(n=9, nk=20, xs=1.0, path='diag_waves/fig_coef_scatter.png'):
	p = first_sol(n)
	coef = band_series(p)
	M = 4096
	x = np.linspace(0, 2 * np.pi, M, endpoint=False)
	Z = coef @ np.exp(1j * np.outer(np.arange(len(coef)), x))
	C = (np.fft.fft(Z) / M)[:nk]
	ks = np.arange(nk)
	cm = plt.cm.turbo(np.linspace(0, 1, nk))

	fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))

	# --- 1. lo pedido: scatter de (Re, Im) de los coeficientes
	a = ax[0]
	for k in ks:
		a.scatter(C[k].real, C[k].imag, color=cm[k], s=170, zorder=3,
				  edgecolor='k', linewidth=.6)
		a.annotate(str(k), (C[k].real, C[k].imag), fontsize=7, ha='center',
				   va='center', color='w', fontweight='bold', zorder=4)
	a.axhline(0, color='#ced4da', lw=1); a.axvline(0, color='#ced4da', lw=1)
	a.set_xlim(-.35, 1.35); a.set_ylim(-.85, .85); a.set_aspect('equal')
	a.set_xlabel('Re $c_k$'); a.set_ylabel('Im $c_k$')
	a.set_title(f'Lo pedido: los {nk} primeros $c_k$ en 2D\n'
				'TODOS caen en Im=0, y valen 0 o 1', fontsize=11)
	a.text(.5, -.62, 'no hay nube: el eje Im está vacío', ha='center',
		   fontsize=10, style='italic', color='#c92a2a')

	# --- 2. lo que sí vive en 2D: los fasores en un x dado
	a = ax[1]
	th = np.linspace(0, 2 * np.pi, 300)
	a.plot(np.cos(th), np.sin(th), color='#ced4da', lw=1)
	occ = [k for k in ks if abs(C[k]) > .5]
	for k in occ:
		v = np.exp(1j * k * xs)
		a.plot([0, v.real], [0, v.imag], color=cm[k], lw=1.6, alpha=.75)
		a.scatter(v.real, v.imag, color=cm[k], s=170, zorder=3,
				  edgecolor='k', linewidth=.6)
		a.annotate(str(k), (v.real, v.imag), fontsize=7, ha='center', va='center',
				   color='w', fontweight='bold', zorder=4)
	a.set_aspect('equal'); a.set_xlabel('Re'); a.set_ylabel('Im')
	a.set_title(f'Los fasores $e^{{ikx}}$ en $x={xs}$\n'
				f'{len(occ)} armónicos ocupados de los primeros {nk}', fontsize=11)

	# --- 3. la cadena de epiciclos que suma a Z(x)
	a = ax[2]
	full_occ = np.nonzero(coef)[0]
	pt = 0 + 0j
	for k in full_occ:
		v = np.exp(1j * k * xs)
		col = plt.cm.turbo(k / len(coef))
		a.plot([pt.real, (pt + v).real], [pt.imag, (pt + v).imag],
			   color=col, lw=2, solid_capstyle='round')
		pt += v
	a.scatter([pt.real], [pt.imag], color='k', s=90, zorder=5, label=f'$Z({xs})$')
	a.scatter([0], [0], color='#868e96', s=50, zorder=5)
	a.plot(Z.real, Z.imag, color='#adb5bd', lw=.6, alpha=.7, zorder=0,
		   label='la hélice completa')
	a.set_aspect('equal'); a.set_xlabel('Re'); a.set_ylabel('Im')
	a.set_title(f'Los {len(full_occ)} epiciclos encadenados\n'
				'la punta traza la hélice', fontsize=11)
	a.legend(fontsize=9)

	fig.suptitle(f'N={n}, solución {p}', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	print(f'coeficientes 1..{nk-1}: '
		  f'{[int(round(v.real)) for v in C[1:nk]]}')
	print(f'max |Im c_k| = {np.abs(C.imag).max():.2e}')
	return path


if __name__ == '__main__':
	print(plot())
