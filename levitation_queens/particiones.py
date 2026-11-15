"""
Descomposiciones EXACTAS de Q(N) en niveles: cada propiedad de una solucion parte el conjunto
de soluciones y da una suma exacta  Q(N) = suma de (cuantas tienen cada valor).

Propiedades:
  primera   : columna de la reina de la fila 0
  cortes    : pasos que no son el alfa dominante
  tramos    : rachas de alfas iguales
  simetria  : tamano de su clase de simetria (8, 4, 2)
  centro    : distancia de la reina de la fila central al centro (solo para ver otra geometria)

Para cada una: los sumandos, y su forma normalizada (fraccion del total), para comparar entre N.

Uso: python particiones.py 10 11 12 13 14
"""
import sys
from collections import Counter
import numpy as np
import min_waves as mw


def props(s, N):
    a = [(s[i + 1] - s[i]) % N for i in range(N - 1)]
    cuts = N - 1 - max(Counter(a).values())
    runs = 1 + sum(a[i] != a[i + 1] for i in range(N - 2))
    return s[0], cuts, runs


if __name__ == '__main__':
    Ns = [int(x) for x in sys.argv[1:]] or [10, 11, 12, 13]
    data = {}
    for N in Ns:
        sols = mw.solutions(N)
        reps = mw.classes(sols, N)
        P = [props(s, N) for s in sols]
        data[N] = {
            'Q': len(sols),
            'primera': Counter(p[0] for p in P),
            'cortes': Counter(p[1] for p in P),
            'tramos': Counter(p[2] for p in P),
            'simetria': Counter(len(m) for m in reps.values()),
        }
    for N in Ns:
        d = data[N]
        print(f"===== N={N}: Q = {d['Q']}")
        for key in ('primera', 'cortes', 'tramos'):
            c = d[key]
            terms = [c[k] for k in sorted(c)]
            print(f"  por {key:8s}: {' + '.join(map(str, terms))} = {sum(terms)}")
        c = d['simetria']
        print(f"  por simetria: " + " + ".join(f"{k}x{c[k]}" for k in sorted(c, reverse=True)) + f" = {sum(k*v for k, v in c.items())}")
    print("\nFORMA normalizada (fraccion del total) -- ¿se repite de un N a otro?")
    print("por columna de la primera reina, en posiciones relativas c/(N-1) = 0, 0.1, ..., 0.5 (es simetrica):")
    for N in Ns:
        c = data[N]['primera']; Q = data[N]['Q']
        f = np.array([c[k] / Q * N for k in range(N)])          # x N para que la media sea 1
        print(f"   N={N}: " + " ".join(f"{np.interp(x*(N-1), range(N), f):5.2f}" for x in (0, .1, .2, .3, .4, .5)))
    print("por cortes, en J/(N-1) = 0.3, 0.4, ..., 0.9:")
    for N in Ns:
        c = data[N]['cortes']; Q = data[N]['Q']
        f = np.array([c.get(k, 0) / Q for k in range(N)])
        print(f"   N={N}: " + " ".join(f"{np.interp(x*(N-1), range(N), f):6.3f}" for x in (.3, .4, .5, .6, .7, .8, .9)))
