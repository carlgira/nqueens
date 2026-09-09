"""
La suma de TODAS las helices de solucion.

  sum_s Z_s(x) = Q * sum_k <c_k> e^{ikx}

El espectro de la suma es el perfil universal (la meseta de maxima entropia).
Como ese perfil es liso, la suma es un PULSO concentrado -- miles de curvas
dentadas se cancelan y queda un unico latido.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import band_series
from diag_waves.orbits import sols

C_ONE, C_ALL = '#adb5bd', '#e8590c'


def plot(n=12, path='diag_waves/fig_aggregate.png'):
	S = sols(n)
	C = np.array([band_series(p) for p in S])
	mean = C.mean(0)
	x = np.linspace(0, 2 * np.pi, 3000)
	E = np.exp(1j * np.outer(np.arange(4 * n), x))
	Zagg = mean @ E
	few = C[np.linspace(0, len(C) - 1, 40).astype(int)] @ E

	fig = plt.figure(figsize=(16.5, 9.5))

	ax = fig.add_subplot(2, 2, 1, projection='3d')
	for z in few[:12]:
		ax.plot(x, z.real, z.imag, color=C_ONE, lw=.6, alpha=.55)
	ax.set_title(f'12 hélices sueltas de N={n}\ndentadas, cada una distinta', fontsize=11)
	ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
	for p_ in (ax.xaxis, ax.yaxis, ax.zaxis): p_.pane.set_visible(False)

	ax2 = fig.add_subplot(2, 2, 2, projection='3d')
	ax2.plot(x, Zagg.real, Zagg.imag, color=C_ALL, lw=1.8)
	ax2.set_title(f'La suma de las {len(S):,} soluciones\n(dividida por Q)', fontsize=11)
	ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([]); ax2.grid(False)
	for p_ in (ax2.xaxis, ax2.yaxis, ax2.zaxis): p_.pane.set_visible(False)

	a3 = fig.add_subplot(2, 2, 3)
	k = np.arange(4 * n)
	a3.stem(k, C[0], linefmt=C_ONE, markerfmt=' ', basefmt=' ', label='una solución (0/1)')
	a3.plot(k, mean, 'o-', color=C_ALL, lw=2, ms=4, label='media de las {:,}'.format(len(S)))
	a3.set_xlabel('armónico $k$'); a3.set_ylabel('$c_k$')
	a3.set_title('Los espectros: picos binarios → meseta lisa', fontsize=11)
	a3.legend(fontsize=9)

	a4 = fig.add_subplot(2, 2, 4)
	a4.plot(x, np.abs(few).T, color=C_ONE, lw=.4, alpha=.35)
	a4.plot(x, np.abs(Zagg), color=C_ALL, lw=2.2, label='la suma / Q')
	a4.set_xlim(0, 2 * np.pi); a4.set_xlabel('x'); a4.set_ylabel('$|Z(x)|$')
	a4.set_title('El módulo: la suma es un PULSO\n'
				 'todo lo demás se cancela', fontsize=11)
	a4.legend(fontsize=9)

	fig.suptitle(f'N={n}: sumar las {len(S):,} hélices de solución da UNA hélice',
				 fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .94))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	w = np.abs(Zagg)
	half = x[np.argmax(w < w.max() / 2)]
	print(f'N={n}: pico |Z|={w.max():.2f} en x=0, cae a la mitad en x={half:.4f} '
		  f'({half/(2*np.pi):.4f} del periodo)')
	return path


if __name__ == '__main__':
	print(plot(12))
