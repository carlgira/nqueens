"""
"Espirales de otro tamaño": ¿una toroidal de p contiene soluciones mas pequeñas que se suman?

Idea del usuario: p = 17 = 11 + 13 - 7. Se escogen k filas de T y se comprimen (las k reinas, con sus
filas y columnas renumeradas 0..k-1 en orden). Si lo que queda es una solucion de k reinas, T
"contiene" una solucion de tamaño k. Dos versiones: normal (diagonales del tablero plano) y toroidal
(diagonales modulo k). Despues: ¿hay A (k1 filas) y B (k2 filas) que cubren todas las filas, cuyo
solape (k1 + k2 - p filas) tambien es solucion?   p = k1 + k2 - k3.

Comparacion: todas las soluciones distintas de p (explicadas y sin explicar) y permutaciones al azar
(no son solucion: miden cuanto se debe solo a ser permutacion).

Uso: ../venv/bin/python tamanos.py p k1 k2
"""
import sys, json
from itertools import combinations
import numpy as np


def comprime(T, filas):
    cols = sorted(T[r] for r in filas); rank = {c: i for i, c in enumerate(cols)}
    return [rank[T[r]] for r in filas]


def valida(U, toro):
    k = len(U)
    if toro:
        return len({(U[i] + i) % k for i in range(k)}) == k and len({(U[i] - i) % k for i in range(k)}) == k
    return len({U[i] + i for i in range(k)}) == k and len({U[i] - i for i in range(k)}) == k


def subsoluciones(T, p, k, toro):
    return [frozenset(R) for R in combinations(range(p), k) if valida(comprime(T, R), toro)]


def sumas(T, p, k1, k2, toro):
    k3 = k1 + k2 - p
    A = subsoluciones(T, p, k1, toro); B = subsoluciones(T, p, k2, toro)
    todo = frozenset(range(p)); n = 0; ejemplo = None
    Bs = set(B)
    for a in A:
        for b in B:
            if a | b != todo: continue
            c = sorted(a & b)
            if valida(comprime(T, c), toro):
                n += 1; ejemplo = ejemplo or (sorted(a), sorted(b), c)
    return len(A), len(B), n, ejemplo


if __name__ == "__main__":
    p, k1, k2 = map(int, sys.argv[1:4])
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    G = {tuple(g["rep"]) for g in json.load(open(f"data/ordenadas_p{p}.json"))}
    from ordenadas import Orbitas
    Ob = Orbitas(p); expl = {Ob.clave(g) for g in G}
    rng = np.random.default_rng(0)
    obj = [(f"{i}:c{o['clase']}x{o['tam']}" + ("" if Ob.clave(o["rep"]) in expl else " SIN EXPLICAR"), o["rep"]) for i, o in enumerate(O)]
    obj += [(f"azar {j}", rng.permutation(p).tolist()) for j in range(5)]
    print(f"p={p} = {k1} + {k2} - {k1 + k2 - p}")
    for toro in (True, False):
        print(f" version {'toroidal' if toro else 'normal'}: nº de sub-soluciones de {k1}, de {k2}, y de parejas (A, B) con solape tambien solucion")
        for nom, T in obj:
            a, b, n, ej = sumas(T, p, k1, k2, toro)
            print(f"   {nom:24s} {a:6d} {b:6d} {n:6d}" + (f"   ej. A={ej[0]} B={ej[1]}" if ej and nom.endswith('EXPLICAR') else ""), flush=True)
