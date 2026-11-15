"""Plano (alpha,beta): a N fijo es un ARREGLO DE RECTAS (Farey), no un fractal.
El refinamiento ocurre al crecer N, no al hacer zoom."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sturmian_plane import conflict_map

fig, ax = plt.subplots(1, 3, figsize=(16, 5.6), facecolor='#0d0d12')
for a, N in zip(ax, (7, 13, 31)):
	M = conflict_map(N, 0, 1, 0, 1, 700)
	cm = ListedColormap(['#00ff9c'] + list(plt.cm.magma(np.linspace(0.05, 0.95, M.max() or 1))))
	a.imshow(M, origin='lower', extent=[0, 1, 0, 1], cmap=cm, vmin=0, vmax=max(M.max(), 1),
	         interpolation='nearest')
	a.set_title(f"N={N}   area con solucion = {(M==0).mean():.1%}", color='#e8e8f0', fontsize=12)
	a.set_xlabel('alpha', color='#8a8a99'); a.tick_params(colors='#8a8a99')
ax[0].set_ylabel('beta', color='#8a8a99')
fig.suptitle("celdas de Farey del plano Sturmiano - verde = solucion exacta", color='#e8e8f0', fontsize=13)
fig.tight_layout(); fig.savefig('fig_sturmian_cells.png', dpi=130, facecolor='#0d0d12')
print('ok')
