import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from spiral_ifs import diag_maps, chaos_game, witness
from tensor import valid


def paint(sig, dim, res=1100, walkers=170_000, steps=80, seed=0):
	lam, t = diag_maps(sig, dim)
	dens, hue, box = chaos_game(lam, t, res=res, walkers=walkers, steps=steps, seed=seed)
	d = np.log1p(dens.astype(float)); d /= d.max() + 1e-12
	img = hsv_to_rgb(np.dstack([(hue + 0.07) % 1.0,
	                            0.92 * np.clip(d * 2.0, 0, 1) ** 0.45,
	                            np.clip(d * 3.2, 0, 1) ** 0.7]))
	return img, box


if __name__ == '__main__':
	N = 7
	cases = [("solucion  3j mod 7", np.array([(3 * j) % 7 for j in range(7)])),
	         ("solucion  2j mod 7", np.array([(2 * j) % 7 for j in range(7)])),
	         ("NO solucion", np.array([0, 1, 3, 5, 2, 6, 4]))]
	fig, ax = plt.subplots(3, 3, figsize=(15.5, 15.5), facecolor='#08080c')
	for row, (name, sig) in zip(ax, cases):
		w = witness(sig)
		for a, dim in zip(row, (1.5, 1.75, 2.0)):
			img, box = paint(sig, dim)
			a.imshow(img, origin='lower', extent=box, interpolation='bilinear')
			a.set_title(f"{name}  dim={dim}   posiciones={w[0]}/7  giros={w[1]}/7   valida={valid(sig)}",
			            color='#e8e8f0', fontsize=9)
			a.set_xticks([]); a.set_yticks([])
	fig.suptitle("f_k(z) = r e^(2pi i (k+s(k))/13) z + e^(2pi i (k-s(k)+6)/13)",
	             color='#e8e8f0', fontsize=13)
	fig.tight_layout(); fig.savefig('fig_spiral_ifs.png', dpi=120, facecolor='#08080c')
	print('grid ok')

	# hero: N=13
	sig13 = np.array([(5 * j) % 13 for j in range(13)])
	assert valid(sig13), witness(sig13)
	img, box = paint(sig13, 1.9, res=1900, walkers=420_000, steps=130, seed=3)
	fig = plt.figure(figsize=(11, 11), facecolor='#08080c')
	a = fig.add_axes([0, 0, 1, 1]); a.set_facecolor('#08080c')
	a.imshow(img, origin='lower', extent=box, interpolation='bilinear')
	a.set_xticks([]); a.set_yticks([])
	fig.savefig('fig_spiral_hero.png', dpi=175, facecolor='#08080c')
	print('hero ok', witness(sig13))
