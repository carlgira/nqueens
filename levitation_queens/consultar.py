"""
Consulta los ficheros data/waves_N{N}.json.

  python consultar.py 8                 resumen de N=8 (histograma de ondas minimas)
  python consultar.py 8 17              ficha de la solucion 17: tablero, ondas, verificacion
  python consultar.py 8 --sol 0 4 7 5 2 6 1 3   busca por la propia solucion
  python consultar.py 8 17 --png        ademas dibuja el campo en fig_N8_17.png
  python consultar.py 8 --fisico        igual, pero con ondas de lambda >= 2 casillas
"""
import argparse, json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(N, fisico=False):
    with open(os.path.join(HERE, 'data', f'waves_N{N}' + ('_fisico' if fisico else '') + '.json')) as fh:
        return json.load(fh)


def field(N, waves, t=0.0):
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    f = np.zeros((N, N))
    for w in waves:
        u, v = w['k']
        f += w['amplitud'] * np.cos(2 * np.pi * (u * I + v * J) / N + w['fase_rad'])
    return f - t


def resumen(d):
    recs = d['soluciones']
    h = Counter(r['ondas_min'] or 0 for r in recs)
    print(f"{'[fisico: lambda >= 2 casillas] ' if d.get('fisico') else ''}N={d['N']}  soluciones={d['n_soluciones']}  clases={d['n_clases']}  delta={d['delta']}")
    print("ondas minimas (en la rejilla) -> n de soluciones:")
    for k in sorted(h):
        print(f"  {k:2d}: {h[k]}" if k else f"  imposible: {h[k]}")
    ne = [r['id'] for r in recs if not r['exacto']]
    print(f"no exactas (limite de tiempo): {len(ne)}" + (f"  ids {ne[:20]}" if ne else ''))
    print(f"con 1 onda libre (k no entero): {sum(r['una_onda_libre'] is not None for r in recs)}")


def ficha(d, r, png=False):
    N = d['N']
    sol = r['solucion']
    print(f"N={N}  solucion #{r['id']}  clase {r['clase']}  {sol}")
    if not r['ondas']:
        print('ninguna combinacion de ondas de este conjunto la levita' if r['exacto'] else 'sin resultado (limite de tiempo)')
        return
    print(f"ondas minimas: {r['ondas_min']}  ({'exacto' if r['exacto'] else 'cota, minimo >= %s' % r['cota_inferior']})"
          f"   1 onda libre: {'k = ' + str(r['una_onda_libre']) if r['una_onda_libre'] is not None else 'no'}")
    print(f"umbral {r['umbral']:.4f}   margen {r['margen']:.4f}")
    print(f"{'k (fila, col)':>14} {'amplitud':>9} {'fase':>8} {'lambda':>8} {'dir':>7}")
    for w in r['ondas']:
        print(f"{str(tuple(w['k'])):>14} {w['amplitud']:9.4f} {w['fase_rad']:8.3f}"
              f" {w['longitud_onda_casillas']:8.3f} {w['direccion_grados']:7.1f}")
    f = field(N, r['ondas'], r['umbral'])
    print("\ncampo (Q = reina, + = por encima del umbral):")
    for i in range(N):
        print('  ' + ' '.join('Q' if sol[i] == j else ('+' if f[i, j] > 0 else '.') for j in range(N)))
    top = set(np.argsort(f.ravel())[-N:])
    ok = top == {i * N + j for i, j in enumerate(sol)}
    print(f"\nverificacion: las {N} casillas mas brillantes son las reinas -> {ok}")
    if png:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        R = 40
        x = np.linspace(-0.5, N - 0.5, N * R)
        I, J = np.meshgrid(x, x, indexing='ij')
        F = np.zeros_like(I)
        for w in r['ondas']:
            u, v = w['k']
            F += w['amplitud'] * np.cos(2 * np.pi * (u * I + v * J) / N + w['fase_rad'])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        from matplotlib.colors import TwoSlopeNorm
        G = F - r['umbral']
        im = ax.imshow(G, cmap='RdBu_r', origin='upper', norm=TwoSlopeNorm(0, G.min(), G.max()),
                       extent=(-0.5, N - 0.5, N - 0.5, -0.5))
        I0, J0 = np.meshgrid(range(N), range(N), indexing='ij')
        ax.scatter(J0.ravel(), I0.ravel(), s=6, c='k', alpha=.35)          # la red de trampas
        ax.scatter(sol, range(N), s=160, facecolors='none', edgecolors='k', lw=1.6)
        ax.set_xticks(range(N)); ax.set_yticks(range(N)); ax.grid(False)
        ax.set_title(f"N={N} #{r['id']}: {r['ondas_min']} ondas")
        fig.colorbar(im, ax=ax, shrink=.8, label='campo - umbral')
        p = os.path.join(HERE, f"fig_N{N}_{r['id']}.png")
        fig.savefig(p, dpi=130, bbox_inches='tight')
        print(f"guardado {p}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('id', type=int, nargs='?')
    ap.add_argument('--sol', type=int, nargs='+')
    ap.add_argument('--png', action='store_true')
    ap.add_argument('--fisico', action='store_true', help='usar el fichero con lambda >= 2 casillas')
    a = ap.parse_args()
    d = load(a.N, a.fisico)
    if a.sol:
        r = next((r for r in d['soluciones'] if r['solucion'] == a.sol), None)
        if r is None:
            raise SystemExit('esa permutacion no es solucion')
        ficha(d, r, a.png)
    elif a.id is not None:
        ficha(d, d['soluciones'][a.id], a.png)
    else:
        resumen(d)
