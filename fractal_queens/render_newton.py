import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from newton_spiral import newton_field, colorize, roots_of
from tensor import valid, toroidal

N = 7
tau = np.array([(3 * j) % N for j in range(N)])           # toroidal, mult c=3
assert valid(tau) and toroidal(tau)

box = (-0.62, 0.62, -0.62, 0.62)
res = 1500
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.7), facecolor='#08080c')
for a_, aa in zip(ax, [1.0, 1.0 + 0.45j, 1.0 + 0.95j]):
	idx, tim, _ = newton_field(roots_of(tau), aa, box, res, iters=110)
	a_.imshow(colorize(idx, tim, N), origin='lower', extent=box, interpolation='bilinear')
	a_.set_title(f"a = {aa:.2f}", color='#e8e8f0', fontsize=12)
	a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("fractal de Newton del polinomio de las reinas (N=7) - las raices SON las reinas",
             color='#e8e8f0', fontsize=13)
fig.tight_layout(); fig.savefig('fig_newton_spiral.png', dpi=140, facecolor='#08080c')
print('fig_newton_spiral.png')
