import math, numpy as np, matplotlib.pyplot as plt
from spiral_queens import SpiralQueens, polar_2_cartesian
sq = SpiralQueens(7); N=7

def points_of(sol):
    theta = np.array([sq.a[v] for v in sol])
    X,Y = polar_2_cartesian(theta, sq.r)
    return np.c_[X,Y]

# 1) una solucion (7 puntos)  2) toda la rejilla (49 puntos, simetria 7-fold)
sol = [int(x) for x in sq.init_sols_label()[14][0]]
one = points_of(sol)
grid = sq.mp.reshape(-1,2)

def structure_factor(pts, kmax=40, ngrid=400):
    kx = np.linspace(-kmax,kmax,ngrid); ky = np.linspace(-kmax,kmax,ngrid)
    KX,KY = np.meshgrid(kx,ky)
    S = np.zeros_like(KX, dtype=complex)
    for x,y in pts:
        S += np.exp(-1j*(KX*x+KY*y))
    return np.abs(S)**2

fig,axs = plt.subplots(2,2,figsize=(9,9))
axs[0,0].scatter(one[:,0],one[:,1],c='k'); axs[0,0].set_title("solucion #14 (7 puntos)")
axs[0,0].set_aspect('equal')
axs[0,1].imshow(structure_factor(one), cmap='inferno', origin='lower')
axs[0,1].set_title("difraccion de la solucion")
axs[1,0].scatter(grid[:,0],grid[:,1],c='k',s=8); axs[1,0].set_title("rejilla completa (49 pts)")
axs[1,0].set_aspect('equal')
axs[1,1].imshow(structure_factor(grid), cmap='inferno', origin='lower')
axs[1,1].set_title("difraccion de la rejilla")
for a in [axs[0,0],axs[1,0]]: a.axis('off')
for a in [axs[0,1],axs[1,1]]: a.axis('off')
fig.suptitle("¿Patron cristalino? Factor de estructura |S(k)|^2")
fig.tight_layout(); fig.savefig("diffraction.png",dpi=130); print("ok")
