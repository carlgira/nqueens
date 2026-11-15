"""
¿Que categorias ("bins") de soluciones se fusionan entre si?

Muestra de soluciones; se cuentan TODAS las fusiones validas A (filas abiertas) + B (cerradas)
entre miembros de la muestra, con todas las puertas de onda, y se reparten por el bin de A y
el de B. Tasa = fusiones validas / (pares posibles x puertas), relativa a la tasa global
(1.0 = como la media; > 1 = pares que conviene probar primero; ~0 = no merece la pena).

Bins candidatos:
  alfa   : |alfa dominante|  (pendiente de fondo)
  cortes : pasos que no son el alfa dominante

Uso: python bins.py 13 --muestra 4000
"""
import argparse, random
from collections import Counter, defaultdict
import numpy as np
import min_waves as mw
from fusion import wave_gates


def feats(s, N):
    a = [(s[i + 1] - s[i]) % N for i in range(N - 1)]
    c = Counter(a)
    dom, n = c.most_common(1)[0]
    dom = min(dom, N - dom)
    return dom, N - 1 - n


def count(L, N, gates, binof):
    full = (1 << N) - 1
    M = Counter()
    for G in gates:
        Gs = sorted(G); Cs = [i for i in range(N) if i not in G]
        index = defaultdict(list)
        partG = []
        for k, s in enumerate(L):
            cm = d1 = d2 = 0
            for i in Cs:
                c = s[i]; cm |= 1 << c; d1 |= 1 << (i + c); d2 |= 1 << (i - c + N)
            index[cm].append((d1, d2, k))
            cg = e1 = e2 = 0
            for i in Gs:
                c = s[i]; cg |= 1 << c; e1 |= 1 << (i + c); e2 |= 1 << (i - c + N)
            partG.append((cg, e1, e2))
        for k, (cg, e1, e2) in enumerate(partG):
            for d1, d2, j in index.get(full ^ cg, ()):
                if j != k and not (e1 & d1) and not (e2 & d2):
                    M[(binof[k], binof[j])] += 1
    return M


def show(M, sizes, ngates, title):
    keys = sorted(sizes)
    tot = sum(M.values())
    npairs = sum(sizes.values()) ** 2 * ngates
    base = tot / npairs
    print(f"\n{title}   (tasa relativa; filas = bin de A, columnas = bin de B; entre parentesis el tamano del bin)")
    print("        " + " ".join(f"{str(k):>7}" for k in keys))
    for a in keys:
        row = []
        for b in keys:
            pairs = sizes[a] * sizes[b] * ngates
            row.append(M.get((a, b), 0) / pairs / base if pairs else 0)
        print(f"{str(a):>4} ({sizes[a]:4d}) " + " ".join(f"{x:7.2f}" for x in row))
    diag = sum(M.get((k, k), 0) for k in keys)
    exp = sum(sizes[k] ** 2 for k in keys) / sum(sizes.values()) ** 2
    print(f"   fusiones dentro del mismo bin: {100*diag/tot:.0f}% (por azar {100*exp:.0f}%)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--muestra', type=int, default=4000)
    a = ap.parse_args()
    N = a.N
    sols = mw.solutions(N)
    L = random.Random(0).sample(sols, min(a.muestra, len(sols)))
    gates = list(wave_gates(N))
    F = [feats(s, N) for s in L]
    for name, col in (('BIN = |alfa dominante|', 0), ('BIN = cortes', 1)):
        binof = [f[col] for f in F]
        M = count(L, N, gates, binof)
        show(M, Counter(binof), len(gates), name)
