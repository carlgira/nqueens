import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import permutations
from diag_waves.wave2d import wave2d

n, M = 7, 13
def collisions(v):
    _, c = np.unique(v % M, return_counts=True); return int((c*(c-1)//2).sum())
def sols(n):
    out, col, dp, dm, cur = [], [False]*n, set(), set(), []
    def rec(r):
        if r == n: out.append(tuple(cur)); return
        for c in range(n):
            if col[c] or (r+c) in dp or (r-c) in dm: continue
            col[c]=True; dp.add(r+c); dm.add(r-c); cur.append(c)
            rec(r+1); col[c]=False; dp.discard(r+c); dm.discard(r-c); cur.pop()
    rec(0); return out

i = np.arange(n)
S = [np.array(p) for p in sols(n)]
A = [np.array(p) for p in permutations(range(n))]
mapS = np.array([[np.mean([collisions(a*i+b*s) for s in S]) for b in range(M)] for a in range(M)])
mapA = np.array([[np.mean([collisions(a*i+b*s) for s in A]) for b in range(M)] for a in range(M)])

good, bad = [int(v) for v in S[3]], [1,3,2,6,4,5,0]
Wg, Wb = np.abs(wave2d(good, M))**2, np.abs(wave2d(bad, M))**2

fig, ax = plt.subplots(2, 2, figsize=(13.5, 12))
def cross(a, lw=1.6):
    t = np.arange(-1, M+1)
    for (dx, dy), col, lab in (((1,1), '#e8590c', '$u=v$  (canal $d^+$)'),
                               ((1,-1), '#0ca678', '$u=-v$  (canal $d^-$)')):
        a.plot(t, (dy*t) % M, '.', color=col, ms=7, label=lab)
    a.axhline(0, color='#868e96', lw=lw, ls=':')
    a.axvline(0, color='#868e96', lw=lw, ls=':')
    a.set_xlim(-.5, M-.5); a.set_ylim(-.5, M-.5)

for a, Z, t in ((ax[0,0], Wg, f'$|W(u,v)|^2$  ·  SOLUCIÓN {good}'),
                (ax[0,1], Wb, f'$|W(u,v)|^2$  ·  NO SOLUCIÓN {bad}')):
    im = a.imshow(Z.T, origin='lower', cmap='magma', vmin=0, vmax=max(Wg.max(), Wb.max()))
    cross(a); a.set_title(t, fontsize=11); a.set_xlabel('u'); a.set_ylabel('v')
    plt.colorbar(im, ax=a, fraction=.046)
ax[0,0].legend(loc='upper right', fontsize=8.5, framealpha=.9)

mS, mA = mapS.copy(), mapA.copy()
mS[0,0] = mA[0,0] = np.nan                       # el origen (todo colisiona) aplasta la escala
vmax = np.nanmax([mS, mA])
for a, Z, t in ((ax[1,0], mS, 'colisiones medias por dirección  ·  las 40 SOLUCIONES'),
                (ax[1,1], mA, 'colisiones medias por dirección  ·  las 5040 permutaciones')):
    im = a.imshow(Z.T, origin='lower', cmap='viridis', vmin=0, vmax=vmax)
    cross(a); a.set_title(t, fontsize=11); a.set_xlabel('a'); a.set_ylabel('b')
    plt.colorbar(im, ax=a, fraction=.046)
print('cruz (soluciones):', np.nanmin(mS), '  fuera de la cruz:',
      np.nanmin([mS[a,b] for a in range(M) for b in range(M)
                 if a and b and b not in (a, (-a) % M)]))

fig.suptitle('La onda 2D: $W(u,v) = \\sum_i e^{2\\pi i(iu + \\sigma(i)v)/M}$,  N=7, M=13\n'
             'cada dirección $(a,b)$ del plano es un canal diagonal generalizado  '
             '$a\\cdot i + b\\cdot\\sigma(i)$', fontsize=13)
fig.tight_layout(rect=(0,0,1,.95))
fig.savefig('diag_waves/fig_wave2d.png', dpi=150, bbox_inches='tight')
print('ok')
