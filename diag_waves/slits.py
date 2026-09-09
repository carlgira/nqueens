"""
N-reinas como experimento de difraccion.

En campo lejano (Fraunhofer) el patron sobre la pantalla es el modulo al
cuadrado de la transformada de la apertura. Si se abren rendijas puntuales en
las posiciones d_1..d_N:

	I(x) = |sum_i exp(i*d_i*x)|^2 = |W(x)|^2

que es EXACTAMENTE el testigo de diag_waves.py. Asi que toda la sesion ha
estado calculando patrones de difraccion sin llamarlos asi. La traduccion:

	rendija            <->  diagonal ocupada
	posicion           <->  indice de la diagonal
	patron en pantalla <->  |W(x)|^2
	intensidad media   <->  N + 2*conflictos          (Parseval)
	franja brillante   <->  una diferencia de diagonales que se repite

Consecuencias, todas verificables:

1. La media de la intensidad sobre la pantalla vale N + 2*conflictos. Una
   SOLUCION es la apertura cuyo patron tiene la MENOR intensidad media
   posible: el patron mas PLANO.

2. I(x) es la transformada de la AUTOCORRELACION de las posiciones. Por eso el
   patron no cambia si se traslada, se refleja o se invierte la apertura --
   que es justo el invariante de simetria de orbit_geo.py. El grupo diedrico
   del tablero no toca el patron de difraccion.

3. Y por eso mismo, dos aperturas distintas con la misma autocorrelacion dan
   el MISMO patron: es el problema de recuperacion de fase. Las 13 parejas de
   orbitas que colisionan en N=9 son parejas de aperturas opticamente
   indistinguibles.

La doble rendija clasica es el caso N=2, d = {0, a}: I(x) = 4cos^2(a x/2).
"""

import numpy as np

from diag_waves.diag_waves import channels, conflicts

GREY, BLUE, ORANGE, PURPLE, GREEN = '#dee2e6', '#1c7ed6', '#e8590c', '#7048e8', '#0ca678'


def screen(positions, x, width=0.0):
	"""
	Intensidad en la pantalla para rendijas en `positions`.

	width > 0 anade el ancho real de cada rendija, que multiplica por la
	envolvente sinc^2 de una sola rendija -- lo que se ve en un laboratorio.
	"""
	p = np.asarray(positions, float)
	I = np.abs(np.exp(1j * np.outer(x, p)).sum(axis=1)) ** 2
	if width > 0:
		I = I * np.sinc(width * x / (2 * np.pi)) ** 2
	return I


def mean_intensity(positions, m=2048):
	"""<I> = N + 2*conflictos, exacto por Parseval."""
	x = 2 * np.pi * np.arange(m) / m
	return float(screen(positions, x).mean())


def board_mask(perm, up=8):
	"""El tablero como mascara 2D de agujeros. Difraccion = |FFT2|^2."""
	n = len(perm)
	M = np.zeros((n * up, n * up))
	for r, c in enumerate(perm):
		M[r * up + up // 2, c * up + up // 2] = 1
	return M


def far_field_2d(M):
	return np.abs(np.fft.fftshift(np.fft.fft2(M))) ** 2


def _mpl():
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	return plt


def plot(n=8, path='diag_waves/fig_slits.png'):
	plt = _mpl()
	sol = [0, 4, 7, 5, 2, 6, 1, 3]					# solucion: 8 rendijas distintas
	few = [0, 1, 2, 4, 3, 7, 6, 5]					# 4 conflictos: 5 rendijas
	many = [3, 2, 1, 0, 7, 6, 5, 4]					# 12 conflictos: 2 rendijas
	x = np.linspace(-np.pi, np.pi, 3000)

	fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))

	# --- 1. la doble rendija clasica, para calibrar
	for a, col in ((2.0, BLUE), (5.0, ORANGE)):
		ax[0, 0].plot(x, screen([0, a], x, width=0.6), color=col, lw=1.6,
					  label=f'separacion a = {a}')
	ax[0, 0].set_title('1. doble rendija clasica\n$I = 4\\cos^2(ax/2)$ por la envolvente', fontsize=10)
	ax[0, 0].legend(fontsize=8); ax[0, 0].set_ylabel('intensidad')

	# --- 2, 3, 4: la misma apertura, con mas y mas rendijas repetidas
	for col_i, (perm, lab) in enumerate(((sol, 'SOLUCION'), (few, 'permutacion con choques'),
										 (many, 'casi todo apilado'))):
		dp = channels(perm)[0]
		u = sorted({int(v) for v in dp})
		c = conflicts(dp)
		a = ax[0, 1 + col_i] if col_i < 2 else ax[1, 0]
		col = (GREEN, BLUE, ORANGE)[col_i]
		a.plot(x, screen(dp, x, width=0.35), color=col, lw=1.2)
		a.axhline(mean_intensity(dp), color='k', ls='--', lw=1)
		a.set_title(f'{2+col_i}. {lab}: {len(u)} rendijas, {c} choques\n'
					f'$\\langle I\\rangle$ = {mean_intensity(dp):.0f} = {n} + 2·{c}', fontsize=10)
		a.text(.02, .95, f'd+ = {u}', transform=a.transAxes, fontsize=7.5,
			   va='top', color='#495057')
	ax[1, 0].set_xlabel('x (angulo en la pantalla)')
	ax[0, 1].set_ylim(0, 70); ax[0, 2].set_ylim(0, 70); ax[1, 0].set_ylim(0, 70)

	# --- 5. el tablero como mascara 2D
	F = far_field_2d(board_mask(sol))
	s_ = F.shape[0] // 2; w = 40
	ax[1, 1].imshow(np.log1p(F[s_ - w:s_ + w, s_ - w:s_ + w]), cmap='inferno')
	ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
	ax[1, 1].set_title('5. el TABLERO como mascara de agujeros\n'
					   'difraccion 2D de una solucion', fontsize=10)

	# --- 6. los dos cortes diagonales de esa figura SON los dos testigos
	m = 2 * n - 1
	t = np.arange(m)
	board = np.zeros((m, m)); board[np.arange(n), np.asarray(sol)] = 1
	W = np.fft.ifft2(board) * m ** 2
	up = np.abs(W[t, t]) ** 2
	dn = np.abs(W[t, (-t) % m]) ** 2
	ax[1, 2].plot(t, up, 'o-', color=GREEN, lw=1.6, label=f'corte u=v  ->  <I>={up.mean():.0f}')
	ax[1, 2].plot(t, dn, 's--', color=PURPLE, lw=1.6, label=f'corte u=-v  ->  <I>={dn.mean():.0f}')
	ax[1, 2].axhline(n, color='k', ls=':', lw=1)
	ax[1, 2].legend(fontsize=8)
	ax[1, 2].set_title('6. los dos cortes DIAGONALES de esa figura\n'
					   'son los dos testigos (media = N si es solucion)', fontsize=10)
	ax[1, 2].set_xlabel('t')

	fig.suptitle(f'N={n}: las diagonales son rendijas, $|W(x)|^2$ es la pantalla', y=.99)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def check(n=8, trials=400, seed=0):
	"""<I> = N + 2*conflictos, y el minimo son las soluciones."""
	rng = np.random.default_rng(seed)
	worst = 0.0
	for _ in range(trials):
		p = rng.permutation(n)
		dp, dm = channels(p)
		worst = max(worst, abs(mean_intensity(dp) - (n + 2 * conflicts(dp))))
	print(f'N={n}  max |<I> - (N + 2*conflictos)| sobre {trials} aperturas: {worst:.2e}')
	print(f'      intensidad media minima posible = {n} (patron perfectamente plano)')


def autocorr(S, L):
	S = sorted(S)
	v = np.zeros(L, int)
	for i in range(len(S)):
		for j in range(i + 1, len(S)):
			v[S[j] - S[i]] += 1
	return v


def plot_interference(n=8, path='diag_waves/fig_slits_interf.png'):
	"""
	Que minimiza exactamente una solucion, y que NO minimiza.

	MINIMIZA la potencia total: <I> = sum_v c_v^2 = N + 2*conflictos. Dos
	reinas en la misma diagonal no son dos rendijas: son UNA con amplitud 2,
	que transmite 4 veces la potencia en vez de 2. La condicion de N-reinas es
	"repartir las N unidades de amplitud lo mas finamente posible".

	NO minimiza las franjas. Con las posiciones ya distintas,

		I(x) = N + 2*sum_{k>=1} A(k) cos(kx),   sum_k A(k) = C(N,2)  FIJO

	asi que la energia de las franjas no depende de la solucion: solo cambia
	COMO se reparte entre los k. Para no tener franjas haria falta A(k) <= 1
	-- un conjunto de Sidon -- y eso es geometricamente imposible aqui: N
	marcas en 2N-1 huecos es densidad 1/2, y un Sidon en M huecos admite
	~sqrt(M) marcas. En N=8: 8 marcas donde caben 4.

	Medido sobre TODAS las aperturas validas de N=8: max A(k) va de 4 a 7.
	Nunca cerca de 1. Las franjas son inevitables.
	"""
	plt = _mpl()
	x = np.linspace(-np.pi, np.pi, 4001)
	sol = sorted({int(v) for v in channels([0, 4, 7, 5, 2, 6, 1, 3])[0]})
	sid = [0, 1, 4, 9, 11]
	reg = list(range(0, 15, 2))
	L = 2 * n

	fig, ax = plt.subplots(2, 3, figsize=(15.5, 7.6))
	for c, (S, name, col) in enumerate((
			(sol, f'solucion de N={n}\n{len(S) if False else len(sol)} marcas en 15 huecos', GREEN),
			(sid, 'Sidon (Golomb)\n5 marcas, sin diferencias repetidas', BLUE),
			(reg, 'rejilla regular\n8 marcas equiespaciadas', ORANGE))):
		I = screen(S, x)
		ax[0, c].plot(x, I, color=col, lw=1.1)
		ax[0, c].axhline(I.mean(), color='k', ls='--', lw=1)
		ax[0, c].set_title(f'{name}\n$\\langle I\\rangle$={I.mean():.1f}  pico={I.max():.0f}  '
						   f'pico/media={I.max()/I.mean():.1f}', fontsize=9)
		A = autocorr(S, L)
		ax[1, c].bar(range(1, L), A[1:], color=col)
		ax[1, c].axhline(1, color='k', ls=':', lw=1.2)
		ax[1, c].set_xlabel('k (distancia entre marcas)')
		ax[1, c].set_title(f'A(k): max = {A[1:].max()}   suma = {A[1:].sum()}', fontsize=9)
	ax[0, 0].set_ylabel('intensidad en la pantalla')
	ax[1, 0].set_ylabel('pares a distancia k')
	fig.suptitle('lo que una solucion minimiza es la POTENCIA, no las franjas '
				 '(la linea punteada es A(k)=1, el ideal de Sidon)', y=.99)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	check()
	plot()
	plot_interference()
