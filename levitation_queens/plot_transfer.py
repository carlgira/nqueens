"""
Figura de data/transfer_N{N}.json: un ejemplo de campo por cada numero de ondas
y el reparto de las soluciones.

Uso: python plot_transfer.py 11   -> fig_transfer_N11.png
"""
import json, os, sys
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))


def fine_field(N, waves, R=30):
    x = np.linspace(-0.5, N - 0.5, N * R)
    I, J = np.meshgrid(x, x, indexing='ij')
    F = np.zeros_like(I)
    for w in waves:
        u, v = w['k']
        F += w['amplitud'] * np.cos(2 * np.pi * (u * I + v * J) / N + w['fase_rad'])
    return F


def main(N):
    d = json.load(open(os.path.join(HERE, 'data', f'transfer_N{N}.json')))
    recs = d['soluciones']
    h = Counter(r['ondas_cota'] or 0 for r in recs)
    Ks = sorted(k for k in h if k)
    fig = plt.figure(figsize=(4.2 * 3, 4.2 * 2))
    for n, K in enumerate(Ks[:5]):
        # el ejemplo con mayor margen de ese K
        r = max((r for r in recs if r['ondas_cota'] == K), key=lambda r: r['margen'])
        ax = fig.add_subplot(2, 3, n + 1)
        G = fine_field(N, r['ondas']) - r['umbral']
        ax.imshow(G, cmap='RdBu_r', norm=TwoSlopeNorm(0, G.min(), G.max()),
                  extent=(-0.5, N - 0.5, N - 0.5, -0.5))
        I0, J0 = np.meshgrid(range(N), range(N), indexing='ij')
        ax.scatter(J0.ravel(), I0.ravel(), s=4, c='k', alpha=.35)
        ax.scatter(r['solucion'], range(N), s=110, facecolors='none', edgecolors='k', lw=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        ks = ', '.join(f"({w['k'][0]:g},{w['k'][1]:g})" for w in r['ondas'])
        ax.set_title(f"{K} onda{'s' if K > 1 else ''}  ·  #{r['id']}  (de N={r['origen']['N']})\nk: {ks}",
                     fontsize=9)
    ax = fig.add_subplot(2, 3, 6)
    labels = [str(k) for k in Ks] + ['ninguno\nsirve']
    vals = [h[k] for k in Ks] + [h.get(0, 0)]
    cols = ['#378ADD'] * len(Ks) + ['#B4B2A9']
    ax.bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('ondas heredadas necesarias (cota superior)')
    ax.set_ylabel('soluciones')
    ax.set_title(f'N={N}: {d["n_soluciones"]} soluciones, {sum(vals[:-1])} levitadas '
                 f'({100 * sum(vals[:-1]) / d["n_soluciones"]:.0f}%)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'N={N} con las ondas encontradas en N=4..10 (rojo = por encima del umbral; círculos = reinas)',
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(HERE, f'fig_transfer_N{N}.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    print(p)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
