"""Figura 1: la torre tensorial tau^(x)k -- autosemejanza exacta."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensor import kron, valid, toroidal

m, c = 7, 3
tau = np.array([(c * j) % m for j in range(m)])

fig, ax = plt.subplots(2, 2, figsize=(11, 11), facecolor='#0d0d12')
p = tau
for k, a in zip(range(1, 5), ax.ravel()):
	if k > 1:
		p = kron(p, tau)
	N = len(p)
	M = np.zeros((N, N), np.uint8)
	M[p, np.arange(N)] = 1
	a.imshow(M, cmap='magma', interpolation='nearest', vmin=0, vmax=1.35)
	a.set_title(f"k={k}   N={N}   valida={valid(p)}", color='#e8e8f0', fontsize=11)
	a.set_xticks([]); a.set_yticks([]); a.set_facecolor('#0d0d12')
fig.suptitle(r"$\tau^{\otimes k}$,  $\tau(j)=3j\ \mathrm{mod}\ 7$  —  cada reina es un tablero entero",
             color='#e8e8f0', fontsize=14)
fig.tight_layout()
fig.savefig('fig_tensor_tower.png', dpi=130, facecolor='#0d0d12')
print('fig_tensor_tower.png')

# zoom: el mismo objeto a tres escalas
p5 = tau.copy()
for _ in range(4):
	p5 = kron(p5, tau)
N = len(p5)
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), facecolor='#0d0d12')
for a, w in zip(ax, [N, N // 7, N // 49]):
	sel = np.arange(w)
	a.scatter(sel, p5[sel] % w if w < N else p5[sel], s=max(1.0, 2600 / w),
	          c='#ffb347', marker='s', linewidths=0)
	a.set_xlim(0, w); a.set_ylim(0, w); a.set_aspect('equal')
	a.set_title(f"ventana {w}x{w}", color='#e8e8f0')
	a.set_xticks([]); a.set_yticks([]); a.set_facecolor('#0d0d12')
fig.suptitle("zoom x7: el bloque es el tablero", color='#e8e8f0', fontsize=13)
fig.tight_layout(); fig.savefig('fig_tensor_zoom.png', dpi=130, facecolor='#0d0d12')
print('fig_tensor_zoom.png')
