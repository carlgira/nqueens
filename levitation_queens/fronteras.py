"""
Fronteras para la busqueda: repartir Q(N) en cajones predecibles.

Datos: data/primeras/N{N}.txt  (./primeras N): cuenta[c0][c1] = soluciones con la reina de la
fila 0 en la columna c0 y la de la fila 1 en c1.

  1 reina : N cajones   (sumas por filas de la matriz)
  2 reinas: ~N^2 cajones

Forma = fraccion del total en coordenadas relativas (c/(N-1)). Prediccion de N a partir de
la forma media de N-1 y N-2, con el total real de N (se mide el REPARTO, no el total).

Uso: python fronteras.py
"""
import os, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    D = {}
    for f in glob.glob(os.path.join(HERE, 'data', 'primeras', 'N*.txt')):
        N = int(os.path.basename(f)[1:-4])
        M = np.loadtxt(f, dtype=np.float64)
        if M.shape == (N, N) and M.sum() > 0:
            D[N] = M
    return D


def shape1(f, x):
    N = len(f)
    return np.interp(x, np.arange(N) / (N - 1), f / f.sum() * N)


def pred1(D, N, srcs):
    x = np.arange(N) / (N - 1)
    sh = np.mean([shape1(D[s].sum(1), x) for s in srcs], axis=0)
    return sh / sh.sum() * D[N].sum()


def shape2(M, X0, DX):
    """densidad en (x0 = c0/(N-1), d = (c1-c0)/(N-1)) por interpolacion bilineal sobre c1-c0"""
    N = len(M)
    out = np.zeros_like(X0)
    c0 = X0 * (N - 1)
    dd = DX * (N - 1)
    for idx in np.ndindex(X0.shape):
        a, d = c0[idx], dd[idx]
        a0 = int(math.floor(a)); a1 = min(a0 + 1, N - 1); ta = a - a0
        d0 = int(math.floor(d)); td = d - d0
        v = 0.0
        for aa, wa in ((a0, 1 - ta), (a1, ta)):
            for de, wd in ((d0, 1 - td), (d0 + 1, td)):
                c1 = aa + de
                if 0 <= c1 < N and wa * wd > 0:
                    v += wa * wd * M[aa, c1]
        out[idx] = v
    return out / M.sum() * N * N


def pred2(D, N, srcs):
    c0, c1 = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    X0 = c0 / (N - 1); DX = (c1 - c0) / (N - 1)
    sh = np.mean([shape2(D[s], X0, DX) for s in srcs], axis=0)
    sh[np.abs(c1 - c0) <= 1] = 0                       # casillas prohibidas (misma columna o diagonal)
    return sh / sh.sum() * D[N].sum()


if __name__ == '__main__':
    D = load()
    Ns = sorted(D)
    print("1) CAJONES POR LA PRIMERA REINA: forma (x N) en c/(N-1) = 0, 0.1, ..., 0.5")
    for N in Ns:
        if N >= 11:
            print(f"   N={N}: " + " ".join(f"{shape1(D[N].sum(1), x):5.3f}" for x in (0, .1, .2, .3, .4, .5)))
    print("\n   prediccion del reparto con la forma media de N-1 y N-2:")
    print("     N | cajones | error medio (ponderado) | error maximo")
    for N in Ns:
        if N - 2 in D and N >= 13:
            p = pred1(D, N, [N - 1, N - 2]); r = D[N].sum(1)
            e = np.abs(p - r) / r
            print(f"    {N:2d} | {N:4d} | {100*np.sum(np.abs(p-r))/r.sum():5.2f}% | {100*e.max():5.1f}%")
    print("\n2) CAJONES POR LAS DOS PRIMERAS REINAS")
    print("     N | cajones no vacios | mayor | menor | error medio (ponderado) | cajones con error < 5% | < 10%")
    for N in Ns:
        if N - 2 in D and N >= 13:
            p = pred2(D, N, [N - 1, N - 2]); r = D[N]
            m = r > 0
            e = np.abs(p[m] - r[m]) / r[m]
            print(f"    {N:2d} | {int(m.sum()):5d} | {int(r.max()):9d} | {int(r[m].min()):8d} | {100*np.abs(p-r).sum()/r.sum():5.2f}% | "
                  f"{100*(e<.05).mean():4.0f}% | {100*(e<.10).mean():4.0f}%")
