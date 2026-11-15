"""
Figura del analisis (python analisis.py primero): fig_analisis.png
  a) ondas vs N: media, moda, maximo y la referencia ceil(N/2)
  b) reparto de direcciones a:b por N (% de las ondas de ese N)
  c) parecido entre N (distribuciones de direcciones)
"""
import json, math, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'          # categorico, slots 1-3
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    A = json.load(open(os.path.join(HERE, 'data', 'analisis.json')))
    Ns = sorted(int(n) for n in A['ondas'])
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK})
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 1], wspace=.32)

    # a) ondas vs N
    ax = fig.add_subplot(gs[0])
    ref = [math.ceil(N / 2) for N in Ns]
    ax.plot(Ns, ref, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(5.5, 3.25, 'ceil(N/2)', color=INK2, va='bottom', ha='center', fontsize=9)
    for key, col, lab in (('max', S1, 'máximo'), ('media', S2, 'media'), ('moda', S3, 'moda')):
        y = [A['ondas'][str(N)][key] for N in Ns]
        ax.plot(Ns, y, color=col, lw=2, marker='o', ms=6, mec='white', mew=1.5, zorder=3)
        ax.text(Ns[-1] + .15, y[-1] + (0 if key == 'max' else (-.25 if key == 'media' else .2)),
                lab, color=INK, va='center', fontsize=9)
    ax.set_xticks(Ns); ax.set_xlim(Ns[0] - .3, Ns[-1] + 1.6); ax.set_ylim(0, 7)
    ax.set_xlabel('N'); ax.set_ylabel('ondas mínimas por solución')
    ax.set_title('a) número de ondas', loc='left', fontsize=11)
    ax.grid(axis='y', color=GRID, lw=.6); ax.spines[['top', 'right']].set_visible(False)
    ax.text(11, 6.55, 'N=11: 6 = cota\nen 448 de 488', fontsize=8, color=INK2, ha='center')

    # b) direcciones
    ax = fig.add_subplot(gs[1])
    tot = {}
    for N in Ns:
        for s, v in A['direcciones'][str(N)].items():
            tot[s] = tot.get(s, 0) + v
    slopes = sorted(tot, key=lambda s: (max(map(int, s.split(':'))), int(s.split(':')[0])))
    slopes = [s for s in slopes if tot[s] >= 20]
    M = np.array([[100 * A['direcciones'][str(N)].get(s, 0) / sum(A['direcciones'][str(N)].values())
                   for s in slopes] for N in Ns])
    im = ax.imshow(M, cmap='Blues', aspect='auto', vmin=0, vmax=60)
    for i in range(len(Ns)):
        for j in range(len(slopes)):
            if M[i, j] >= 1:
                ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if M[i, j] > 35 else INK)
    ax.set_xticks(range(len(slopes)))
    ax.set_xticklabels([f"{s}\n{math.degrees(math.atan2(*map(int, s.split(':')))):.0f}°" for s in slopes], fontsize=8)
    ax.set_yticks(range(len(Ns))); ax.set_yticklabels(Ns)
    ax.set_ylabel('N'); ax.set_xlabel('dirección a:b  (ordenadas por max(a,b); ángulo plegado a 0–45°)')
    ax.set_title('b) reparto de direcciones, % de las ondas de cada N', loc='left', fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # c) similitud
    ax = fig.add_subplot(gs[2])
    Sm = np.array([[A['similitud'][f'{a},{b}'] for b in Ns] for a in Ns])
    ax.imshow(Sm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(Ns)):
        for j in range(len(Ns)):
            ax.text(j, i, f'{Sm[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                    color='white' if Sm[i, j] > .6 else INK)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
    ax.set_yticks(range(len(Ns))); ax.set_yticklabels(Ns)
    ax.set_title('c) parecido de direcciones entre N', loc='left', fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)

    p = os.path.join(HERE, 'fig_analisis.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor='white')
    print(p)


if __name__ == '__main__':
    main()
