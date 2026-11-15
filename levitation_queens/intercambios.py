"""
¿Cuantas reinas hay que mover para pasar de una toroidal a otra?

Para cada solucion distinta (data/orbitas_p{p}.json) se liberan k filas, de todas las formas, y se
buscan todas las maneras de recolocar esas k reinas que dan otra toroidal. Se anota a que solucion
(salvo simetria) se llega. Resultado: el grafo de "vecinas a k reinas" entre las soluciones distintas.

Uso: ../venv/bin/python intercambios.py p kmax
"""
import sys, json
from itertools import combinations
from collections import defaultdict
from orbitas import mats, apply
from haces import canon


def indice(p, O):
    Ms = mats(p); idx = {}
    for i, o in enumerate(O):
        for M in Ms:
            idx[canon([int(x) for x in apply(M, o["rep"], p)], p)] = i
    return idx


def vecinas(T, p, k):
    """todas las toroidales que difieren de T en exactamente k filas."""
    out = []
    for R in combinations(range(p), k):
        cols = [T[r] for r in R]
        fs = {(T[r] + r) % p for r in R}; fd = {(T[r] - r) % p for r in R}   # diagonales liberadas
        U = list(T)
        def rec(i, usados, sm, df):
            if i == k:
                out.append(tuple(U)); return
            r = R[i]
            for j, c in enumerate(cols):
                if usados >> j & 1 or c == T[r]: continue
                s, d = (c + r) % p, (c - r) % p
                if s not in fs or d not in fd or s in sm or d in df: continue
                U[r] = c
                rec(i + 1, usados | 1 << j, sm | {s}, df | {d})
            U[r] = T[r]
        rec(0, 0, frozenset(), frozenset())
    return out


if __name__ == "__main__":
    p, kmax = int(sys.argv[1]), int(sys.argv[2])
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    idx = indice(p, O)
    nombre = [f"{i}:c{o['clase']}x{o['tam']}" for i, o in enumerate(O)]
    dist = {}                                   # (i, j) -> k minimo
    for k in range(3, kmax + 1):
        print(f"k={k}", flush=True)
        for i, o in enumerate(O):
            hacia = defaultdict(int)
            for U in vecinas(o["rep"], p, k):
                hacia[idx[canon(list(U), p)]] += 1
            for j in hacia: dist.setdefault((i, j), k)
            print(f"   {nombre[i]:12s} -> " + (", ".join(f"{nombre[j]}({n})" for j, n in sorted(hacia.items())) or "nada"),
                  flush=True)
    # componentes del grafo con todos los k probados
    padre = list(range(len(O)))
    def f(x):
        while padre[x] != x: x = padre[x]
        return x
    for (i, j) in dist: padre[f(i)] = f(j)
    comp = defaultdict(list)
    for i in range(len(O)): comp[f(i)].append(nombre[i])
    print("componentes:", list(comp.values()))
