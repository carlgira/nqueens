"""
Relacion entre clases: ¿una toroidal de clase m sale de otra de clase menor cambiando solo las
reinas de UNA recta de su cobertura minima?
Para cada toroidal S (un representante por orbita de traslacion) y cada recta L de cada cobertura
minima: se liberan las filas que solo cubre L y se buscan todas las demas toroidales S' que coinciden
con S en el resto de filas. Se anota la clase de S'.
Modelo de azar: lo mismo liberando un conjunto de filas al azar del mismo tamano.
Uso: python vecinos.py p [max_coberturas]
"""
import sys, random
from collections import Counter, defaultdict
from orbitas import load, min_covers, canon_tr

def completar(T, libres, p):
    """todas las toroidales que coinciden con T fuera de 'libres'."""
    fijo = [r for r in range(p) if r not in libres]
    cols = {T[r] for r in fijo}; s = {(T[r] + r) % p for r in fijo}; d = {(T[r] - r) % p for r in fijo}
    U = list(T); out = []; L = sorted(libres)
    def rec(i):
        if i == len(L): out.append(tuple(U)); return
        r = L[i]
        for c in range(p):
            if c in cols or (c + r) % p in s or (c - r) % p in d: continue
            cols.add(c); s.add((c + r) % p); d.add((c - r) % p); U[r] = c
            rec(i + 1)
            cols.discard(c); s.discard((c + r) % p); d.discard((c - r) % p)
    rec(0)
    return [u for u in out if u != tuple(T)]

if __name__ == '__main__':
    p = int(sys.argv[1]); maxcov = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    random.seed(1)
    reps = load(p)
    clase = lambda U: reps[canon_tr(U, p)]
    real = defaultdict(Counter); nulo = defaultdict(Counter); n = Counter()
    for T, m in sorted(reps.items()):
        if m < 2: continue
        k, covs = min_covers(T, p)
        conj = set()
        for cv in covs[:maxcov]:
            for i, (a, c, mask) in enumerate(cv):
                otros = 0
                for j, l in enumerate(cv):
                    if j != i: otros |= l[2]
                conj.add(mask & ~otros)
        n[m] += 1
        vistas = set(); vistas0 = set()
        for mask in conj:
            libres = {r for r in range(p) if mask >> r & 1}
            for U in completar(T, libres, p): vistas.add(clase(U))
            lib0 = set(random.sample(range(p), len(libres)))
            for U in completar(T, lib0, p): vistas0.add(clase(U))
        for c in vistas: real[m][c] += 1
        for c in vistas0: nulo[m][c] += 1
        real[m]['alguna'] += bool(vistas); nulo[m]['alguna'] += bool(vistas0)
        real[m]['menor'] += any(c < m for c in vistas); nulo[m]['menor'] += any(c < m for c in vistas0)
        real[m]['m-1'] += (m - 1) in vistas; nulo[m]['m-1'] += (m - 1) in vistas0
    print(f"p={p}: orbitas de traslacion con al menos una vecina (cambiando las filas de una recta) | azar (mismo numero de filas, elegidas al azar)")
    for m in sorted(n):
        f = lambda d, key: f"{d[m][key]}/{n[m]}"
        print(f"  clase {m}: vecina de clase m-1: {f(real,'m-1')} | {f(nulo,'m-1')};  de clase menor: {f(real,'menor')} | {f(nulo,'menor')};  alguna: {f(real,'alguna')} | {f(nulo,'alguna')}")
        print(f"       clases de las vecinas: real { {c: v for c, v in sorted(real[m].items(), key=str) if isinstance(c, int)} }  azar { {c: v for c, v in sorted(nulo[m].items(), key=str) if isinstance(c, int)} }")
