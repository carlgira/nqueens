"""
¿Una solucion compleja es una COMBINACION de toroidales simples?

cobertura(sigma) = minimo numero de toroidales de un tramo (a*i + c, pendientes cualesquiera)
cuya union contiene las N reinas de sigma. Cada fila toma su reina de una de ellas.

  1 -> la propia solucion es toroidal de un tramo
  2 -> mezcla de dos toroidales (cada fila elige una u otra)
  ...
Calculo exacto: por la fila 0 pasan pocas toroidales compatibles (una por pendiente), asi
que se ramifica sobre "la primera fila sin cubrir" con profundidad creciente.

Uso: python combinaciones.py 7 11 13
"""
import json, os, sys, random
from collections import Counter, defaultdict
import numpy as np
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))


def slopes(N):
    import math
    return [a for a in range(2, N - 1) if math.gcd(a, N) == 1 and math.gcd(a + 1, N) == 1 and math.gcd(a - 1, N) == 1]


def cover(sol, N, SL):
    """-> (minimo, lista de (pendiente, desplazamiento, filas cubiertas))"""
    full = (1 << N) - 1
    # mascara de filas de sol que caen en la toroidal (a, c)
    masks = {}
    for a in SL:
        by = defaultdict(int)
        for i, x in enumerate(sol):
            by[(x - a * i) % N] |= 1 << i
        masks[a] = by
    best = [None]

    def rec(covered, used, depth):
        if covered == full:
            best[0] = list(used)
            return True
        if depth == 0:
            return False
        i = 0
        while covered >> i & 1:
            i += 1
        opts = sorted(((masks[a][(sol[i] - a * i) % N], a) for a in SL), key=lambda t: -bin(t[0] & ~covered).count('1'))
        for m, a in opts:
            used.append((a, (sol[i] - a * i) % N, m))
            if rec(covered | m, used, depth - 1):
                return True
            used.pop()
        return False

    for k in range(1, N + 1):
        if rec(0, [], k):
            return k, best[0]


if __name__ == '__main__':
    for N in [int(x) for x in sys.argv[1:]] or [7, 11, 13]:
        SL = slopes(N)
        sols = mw.solutions(N)
        if len(sols) > 20000:
            sols = random.Random(0).sample(sols, 20000)
        K = {}
        f = os.path.join(HERE, 'data', f'waves_N{N}_nodeg.json')
        if os.path.exists(f):
            K = {tuple(r['solucion']): r['ondas_min'] for r in json.load(open(f))['soluciones']}
        c = Counter(); byc = defaultdict(list); sizes = defaultdict(Counter)
        cv = {}
        for s in sols:
            k, used = cover(s, N, SL)
            cv[s] = k; c[k] += 1
            if K:
                byc[k].append(K[s])
            sizes[k][tuple(sorted((bin(m).count('1') for _, _, m in used), reverse=True))] += 1
        print(f"N={N}: {len(sols)} soluciones{' (muestra)' if len(sols) == 20000 else ''}, {len(SL)} pendientes toroidales")
        for k in sorted(c):
            extra = f" | ondas (media) {np.mean(byc[k]):.2f}" if K else ''
            print(f"   {k} toroidales: {c[k]:6d} ({100*c[k]/len(sols):4.1f}%){extra} | reparto de reinas mas comun: {dict(sizes[k].most_common(3))}")
        if K:
            print(f"   correlacion con ondas: {np.corrcoef([cv[s] for s in sols], [K[s] for s in sols])[0,1]:.2f}")
