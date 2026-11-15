"""
Quita duplicados del censo de haces: soluciones que admiten mas de un haz n+m.

./haces cuenta cada pareja (solucion, haz) una vez. Para cada solucion base se cuentan TODOS los haces
n+m que admite (cualquier pareja de pendientes): h. Orbitas de traslacion distintas = suma de mult/h,
con mult = (p-1)/(n(n-1)m). Validacion: p=17 3+1 -> 64 y p=23 3+1 -> 88 (conocidos por otra via).

Uso: ../venv/bin/python haces_distintas.py p n m
"""
import sys
import numpy as np
from collections import Counter
from fractions import Fraction


def n_haces(T, p, n, m, top):
    """numero de haces n+m que admite T. top[a] = offsets de pendiente a ordenados por nº de reinas."""
    h = 0; r = np.arange(p)
    for a in range(2, p - 1):
        oa, ca = top[a]
        if len(oa) < n: continue
        for b in range(2, p - 1):
            if a == b or (n == m and a > b): continue
            ob, cb = top[b]
            if len(ob) < m or ca[:n].sum() + cb[:m].sum() < p: continue
            enA = [(T - a * r) % p == c for c in oa[:n]]; enB = [(T - b * r) % p == c for c in ob[:m]]
            tot = np.sum(enA + enB, axis=0)
            if tot.min() == 0: continue
            if all((e & (tot == 1)).any() for e in enA + enB): h += 1
    return h


if __name__ == "__main__":
    p, n, m = map(int, sys.argv[1:4])
    D = np.loadtxt(f"data/haces_base_p{p}_{n}_{m}.txt", dtype=np.int64, ndmin=2)
    S = D[:, 2:]; N = len(S); r = np.arange(p)
    # cribado rapido: para cada pendiente, cuantas reinas tienen las n (o m) rectas mas cargadas
    mejor = np.zeros((p, N, max(n, m)), dtype=np.int64)
    for a in range(2, p - 1):
        off = (S - a * r) % p
        cnt = np.zeros((N, p), dtype=np.int64)
        np.add.at(cnt, (np.repeat(np.arange(N), p), off.ravel()), 1)
        cnt.sort(axis=1)
        mejor[a] = cnt[:, ::-1][:, :max(n, m)]
    sA = mejor[:, :, :n].sum(axis=2); sB = mejor[:, :, :m].sum(axis=2)       # [pendiente, solucion]
    mult = Fraction(p - 1, n * (n - 1) * m); total = Fraction(0); hs = Counter()
    for t in range(N):
        # parejas candidatas distintas de la propia
        cand = 0
        a_ok = np.flatnonzero(sA[:, t] + sB[:, t].max() >= p)
        if len(a_ok) <= 1 and n != m and (sA[:, t][:, None] + sB[:, t][None, :] >= p).sum() == 1:
            h = 1
        else:
            T = S[t]; top = {}
            for a in range(2, p - 1):
                c = Counter(((T - a * r) % p).tolist()); o = sorted(c, key=lambda x: -c[x])
                top[a] = (o, np.array([c[x] for x in o]))
            h = n_haces(T, p, n, m, top)
        hs[h] += 1; total += mult / max(h, 1)
    print(f"p={p} haz {n}+{m}: {N} base = {N * mult} orbitas con multiplicidad -> {total} orbitas de traslacion distintas;"
          f"  nº de haces por solucion: {dict(sorted(hs.items()))}")
