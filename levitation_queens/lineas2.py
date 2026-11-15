"""
Clase enumerable "m rectas": soluciones cuyas reinas caen todas sobre m rectas toroidales
(a_k, c_k), a_k en 2..p-2, sin exigir plantilla de onda ni niveles. Aqui m = 2, exhaustivo.

Para cada par de rectas, cada fila elige una de las dos; busqueda con poda.
Se compara con lo que la fusion por niveles alcanza (genealogia_N13.npz) y entre primos.

Uso: python lineas2.py 11 13 17 19 23
"""
import sys, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def two_lines(p):
    lines = [(a, c) for a in range(2, p - 1) for c in range(p)]
    found = {}
    for x in range(len(lines)):
        a1, c1 = lines[x]
        A = [(a1 * i + c1) % p for i in range(p)]
        for y in range(x + 1, len(lines)):
            a2, c2 = lines[y]
            B = [(a2 * i + c2) % p for i in range(p)]
            out = []

            def rec(i, cols, d1, d2, t, used):
                if i == p:
                    if used == 3:
                        out.append(tuple(t))
                    return
                for k, v in ((1, A[i]), (2, B[i])):
                    if k == 2 and v == A[i]:
                        continue
                    b = 1 << v; e = 1 << (i + v); g = 1 << (i - v + p)
                    if cols & b or d1 & e or d2 & g:
                        continue
                    t.append(v); rec(i + 1, cols | b, d1 | e, d2 | g, t, used | k); t.pop()
            rec(0, 0, 0, 0, [], 0)
            for t in out:
                if any((t[i + 1] - t[i]) % p != (t[1] - t[0]) % p for i in range(p - 1)):
                    found.setdefault(t, (a1, a2))
    return found


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]] or [11, 13]:
        F = two_lines(p)
        rel = Counter()
        for t, (a1, a2) in F.items():
            rel['opuestas' if (a1 + a2) % p == 0 else 'misma pendiente' if a1 == a2 else
                'inversas' if a1 * a2 % p in (1, p - 1) else 'otra'] += 1
        print(f"p={p}: {len(F)} soluciones no toroidales sobre 2 rectas | relacion de pendientes: {dict(rel)}", flush=True)
        f = os.path.join(HERE, 'data', f'genealogia_N{p}.npz')
        if os.path.exists(f):
            d = np.load(f)
            lev = {tuple(s): int(l) for s, l in zip(d['S'].tolist(), d['level'])}
            c = Counter(lev[t] for t in F)
            print(f"      nivel de fusion en que aparecen: {dict(sorted(c.items()))}")
