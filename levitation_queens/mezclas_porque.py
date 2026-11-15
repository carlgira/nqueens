"""
¿Por que las mezclas de dos toroidales tienen esos tamaños (9+8, 13+4 en p=17)?

Para cada solucion T que es mezcla A (pieza grande) + B (pieza pequeña D = filas donde T != A):
 1. D tiene que ser cerrada: en esas filas B usa las mismas columnas y las mismas diagonales que A.
    Las piezas posibles son uniones de las orbitas del grupo generado por las tres permutaciones
    de filas "misma columna", "misma diagonal /", "misma diagonal \\" entre A y B.
 2. ¿D es un bloque (clase de mu4 o mu8) alrededor de alguna fila centro?
 3. ¿D son las reinas de una sola recta de la cobertura minima de T?

Uso: ../venv/bin/python mezclas_porque.py 17
"""
import sys, json
import numpy as np
from collections import Counter
from mezclas import carga, mezcla
from orbitas import min_covers
from clase_m import subgrupo


def orbitas_cierre(A, B, p):
    """orbitas (de filas) del grupo que obliga a cambiar filas juntas al pasar de A a B."""
    padre = list(range(p))
    def f(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]; x = padre[x]
        return x
    for g in (lambda T, r: T[r], lambda T, r: (T[r] + r) % p, lambda T, r: (T[r] - r) % p):
        donde = {g(A, r): r for r in range(p)}
        for r in range(p):
            padre[f(r)] = f(donde[g(B, r)])          # la fila de A que tiene lo que B pone en r
    return sorted(Counter(f(r) for r in range(p)).values())


def bloque(D, p):
    """(centro, tamaño del subgrupo) si D - centro es una clase de un subgrupo multiplicativo."""
    for c in range(p):
        X = sorted((r - c) % p for r in D)
        if 0 in X: continue
        H = {x * pow(X[0], p - 2, p) % p for x in X}
        if len(subgrupo(H, p)) == len(H): return c, len(H)
    return None


if __name__ == "__main__":
    p = int(sys.argv[1])
    cl, S = carga(p); cmax = int(cl.max())
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    for i, o in enumerate(O):
        T = o["rep"]
        r = mezcla(T, p, cl, S, cmax)
        if r is None or o["clase"] == 1: continue
        c, a, b, n, m, sel = r
        A = S[np.flatnonzero(sel & (m == a))[0]].tolist(); B = S[np.flatnonzero(sel & (m == b))[0]].tolist()
        D = [x for x in range(p) if T[x] != A[x]]
        k, covs = min_covers(T, p)
        una = any(any(set(D) <= {x for x in range(p) if mk >> x & 1} for _, _, mk in cv) for cv in covs)
        igual = any(any(set(D) == {x for x in range(p) if mk >> x & 1 and sum(m2 >> x & 1 for _, _, m2 in cv) == 1}
                        for _, _, mk in cv) for cv in covs)
        print(f"  {i}:c{o['clase']}x{o['tam']:<4d} pieza pequeña {len(D)} filas {D}: "
              f"bloque {bloque(D, p)} · orbitas A-B {orbitas_cierre(A, B, p)} · "
              f"en una sola recta de T: {'si' if una else 'no'}{' (toda la recta)' if igual else ''}")
