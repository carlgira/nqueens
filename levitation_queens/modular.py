"""
Los angulos viven en la recta proyectiva modulo N, no en el plano euclideo.

Cada onda k = (u, v) genera la recta modular {t*k mod N}. Para N primo, una recta
es una pendiente s = v/u mod N (o infinito). Las 4 direcciones de ataque son
s = 0, inf, 1, -1 (fila, columna, diagonales); las 8 simetrias del tablero actuan
sobre s como  s -> -s, 1/s, -1/s.  Una onda = (clase de recta, armonico t).
El angulo euclideo que se ve (1:2, 2:5, ...) es el alias mas corto de t*(1, s).

Este script:
  1. agrupa todas las ondas usadas por clase de recta y armonico,
  2. compara con la formula de cuantas clases hay para N primo:
        (N-3)/4  si N = 3 mod 4,     (N-1)/4  si N = 1 mod 4,
  3. dibuja el plano k de N=7 y N=11 coloreado por clase (fig_modular.png).

Uso: python modular.py
"""
import math, os
from collections import Counter, defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analisis import load, slope, isprime

HERE = os.path.dirname(os.path.abspath(__file__))
LS = [(1, 0, 0, 1), (0, 1, -1, 0), (-1, 0, 0, -1), (0, -1, 1, 0),
      (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0)]
COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def units(N):
    return [t for t in range(1, N) if math.gcd(t, N) == 1]


def line_key(k, N):
    u, v = k
    return min(((t * (a * u + b * v)) % N, (t * (c * u + d * v)) % N)
               for t in units(N) for a, b, c, d in LS)


def degenerate(k, N):
    u, v = k
    return 0 in (u % N, v % N, (u + v) % N, (u - v) % N)


def predicted_classes(p):
    return (p - 3) // 4 if p % 4 == 3 else (p - 1) // 4


def shortest(k, N):
    return tuple(x if x <= N // 2 else x - N for x in (k[0] % N, k[1] % N))


def main():
    print(f"{'N':>3} {'primo':>6} {'clases usadas':>14} {'clases no degeneradas':>22} {'formula':>8}   dir. euclideas")
    for N in range(4, 12):
        d = load(N)
        used = Counter(line_key(w['k'], N) for r in d for w in r['ondas'])
        nd = {line_key((u, v), N) for u in range(N) for v in range(N)
              if (u, v) != (0, 0) and not degenerate((u, v), N)}
        eu = len({slope(*w['k'])[:2] for r in d for w in r['ondas']})
        usednd = sum(1 for k in used if k in nd)
        f = predicted_classes(N) if isprime(N) and N > 3 else '-'
        print(f"{N:3d} {'si' if isprime(N) else 'no':>6} {usednd:6d} (+{len(used)-usednd} degen.) {len(nd):14d} {str(f):>12}   {eu}")
    print("prediccion para primos siguientes:",
          {p: predicted_classes(p) for p in (13, 17, 19, 23, 29, 31)})

    # figura: plano k (alias mas corto) de N=7 y N=11
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.3))
    for ax, N in zip(axs, (7, 11)):
        d = load(N)
        use = Counter()
        for r in d:
            for w in r['ondas']:
                use[shortest(w['k'], N)] += 1
                use[shortest((-w['k'][0], -w['k'][1]), N)] += 1
        keys = sorted({line_key((u, v), N) for u in range(N) for v in range(N)
                       if (u, v) != (0, 0) and not degenerate((u, v), N)})
        cidx = {k: i for i, k in enumerate(keys)}
        h = N // 2
        for u in range(-h, h + 1):
            for v in range(-h, h + 1):
                if (u, v) == (0, 0):
                    ax.scatter(v, u, s=40, c=INK, marker='+', zorder=3)
                    continue
                if degenerate((u, v), N):
                    ax.scatter(v, u, s=14, c=GRID, zorder=2)
                    continue
                c = COLS[cidx[line_key((u, v), N)]]
                n = use.get((u, v), 0)
                ax.scatter(v, u, s=18 + 260 * n / max(use.values()), c=c, edgecolors='white', lw=1, zorder=3)
        # las 4 direcciones de ataque (degeneradas)
        for (a, b) in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ax.plot([-h * b, h * b], [-h * a, h * a], color=GRID, lw=1, zorder=1)
        ax.set_aspect('equal'); ax.set_xlim(-h - .7, h + .7); ax.set_ylim(-h - .7, h + .7)
        ax.set_xlabel('k₁ (ciclos por tablero, columnas)'); ax.set_ylabel('k₀ (filas)')
        ax.set_title(f'N={N}: {len(keys)} clase{"s" if len(keys) > 1 else ""} de recta modular', loc='left', fontsize=11)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for i, k in enumerate(keys):
            ax.text(1.0, 1.02 + .055 * (len(keys) - 1 - i), f'● clase {k}', color=COLS[i], fontsize=9,
                    va='bottom', ha='right', transform=ax.transAxes)
    fig.suptitle('Cada punto es una onda k posible; tamaño = cuántas veces se usa; gris = degenerada (fila, columna, diagonal)',
                 fontsize=10, color=INK2)
    p = os.path.join(HERE, 'fig_modular.png')
    fig.savefig(p, dpi=130, bbox_inches='tight', facecolor='white')
    print(p)


if __name__ == '__main__':
    main()
