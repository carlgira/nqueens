"""
Genealogia completa desde las toroidales de un tramo (N primo), aplanada:

Toda solucion alcanzada por fusion es "a trozos": cada fila i esta sobre una recta toroidal
    T(i) = a_k * i + c_k  (mod p)        (la recta de la hoja de la que viene esa fila)
Se guarda para cada solucion: nivel, las dos piezas, la plantilla, y la recta de cada fila.
Cuando hay varias formas de obtenerla en el mismo nivel, se queda la de menos rectas distintas
(no garantiza el minimo absoluto).

Uso: python genealogia.py 13      -> data/genealogia_N13.npz
"""
import sys, os, time
from collections import Counter
import numpy as np
import min_waves as mw
from fusion import wave_gates

HERE = os.path.dirname(os.path.abspath(__file__))


def nlines(L):
    s = np.sort(L, axis=1)
    return 1 + (s[:, 1:] != s[:, :-1]).sum(1)


def run(p):
    sols = mw.solutions(p); Qn = len(sols)
    S = np.array(sols, dtype=np.int64)
    idx = {s: i for i, s in enumerate(sols)}
    maps = mw.board_maps(p)
    sym = np.array([[idx[mw.apply_map(g, s)] for g in maps] for s in sols])
    # transformacion de rectas (a, c) -> id a*p+c bajo cada simetria, y de filas
    LT = np.zeros((8, p * p), dtype=np.int64)
    for gi, ((A, B, C, D), (e, f)) in enumerate(maps):
        for a in range(p):
            for c in range(p):
                i0, j0 = (B * c + e) % p, (D * c + f) % p
                i1, j1 = (A + B * (a + c) + e) % p, (C + D * (a + c) + f) % p
                if (i1 - i0) % p == 0:
                    continue
                a2 = (j1 - j0) * pow(int(i1 - i0), -1, p) % p
                LT[gi, a * p + c] = a2 * p + (j0 - a2 * i0) % p
    rng = np.random.default_rng(0)
    W = rng.integers(1, 2 ** 40, size=p, dtype=np.int64)
    X = (S + 1) * W
    gates = [np.isin(np.arange(p), list(G)) for G in wave_gates(p)]
    level = -np.ones(Qn, dtype=np.int64)
    L = np.zeros((Qn, p), dtype=np.int64)
    pa = -np.ones(Qn, dtype=np.int64); pb = pa.copy(); pg = pa.copy()
    for a in range(2, p - 1):
        for c in range(p):
            i = idx[tuple((a * r + c) % p for r in range(p))]
            level[i] = 0; L[i] = a * p + c
    nl = nlines(L)
    lev = 0; t0 = time.time()
    while True:
        lev += 1
        known = np.where(level >= 0)[0]; unk = np.where(level < 0)[0]
        if not len(unk):
            break
        best = np.full(Qn, 99); bl = np.zeros((Qn, p), dtype=np.int64)
        ba = -np.ones(Qn, dtype=np.int64); bb = ba.copy(); bg = ba.copy()
        for gi, m in enumerate(gates):
            hA = X[:, m].sum(1); hB = X[:, ~m].sum(1)
            reps = []
            for h in (hA, hB):
                o = known[np.lexsort((nl[known], h[known]))]
                u, first = np.unique(h[o], return_index=True)
                reps.append((u, o[first]))
            (uA, rA), (uB, rB) = reps
            qa = np.searchsorted(uA, hA[unk]); qa[qa == len(uA)] = 0
            qb = np.searchsorted(uB, hB[unk]); qb[qb == len(uB)] = 0
            ok = (uA[qa] == hA[unk]) & (uB[qb] == hB[unk])
            if not ok.any():
                continue
            T = unk[ok]; a_ = rA[qa[ok]]; b_ = rB[qb[ok]]
            Ln = np.where(m, L[a_], L[b_]); n = nlines(Ln)
            imp = n < best[T]
            T = T[imp]; best[T] = n[imp]; bl[T] = Ln[imp]; ba[T] = a_[imp]; bb[T] = b_[imp]; bg[T] = gi
        direct = np.where(best < 99)[0]
        if not len(direct):
            break
        for t in direct:                               # las 8 simetrias de cada nueva
            for gi, ((A, B, C, D), (e, f)) in enumerate(maps):
                j = sym[t, gi]
                if level[j] >= 0 or best[j] <= best[t]:
                    continue
                rows = (A * np.arange(p) + B * S[t] + e) % p
                Lj = np.zeros(p, dtype=np.int64); Lj[rows] = LT[gi, bl[t]]
                best[j] = best[t]; bl[j] = Lj; ba[j] = t; bb[j] = -1; bg[j] = -1 - gi
        new = np.where(best < 99)[0]
        level[new] = lev; L[new] = bl[new]; nl[new] = best[new]; pa[new] = ba[new]; pb[new] = bb[new]; pg[new] = bg[new]
        print(f"   nivel {lev}: +{len(new)} -> {(level >= 0).sum()}  ({time.time()-t0:.0f}s)", flush=True)
    # comprobacion: cada fila esta en su recta
    r = np.arange(p)
    ok = level >= 0
    assert (((L[ok] // p) * r + L[ok] % p) % p == S[ok]).all()
    np.savez_compressed(os.path.join(HERE, 'data', f'genealogia_N{p}.npz'), S=S, level=level, L=L, nl=nl, pa=pa, pb=pb, pg=pg)
    return S, level, L, nl


if __name__ == '__main__':
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    S, level, L, nl = run(p)
    sg = lambda x: x if x <= p // 2 else x - p
    nsl = np.array([len({abs(sg(int(v) // p)) for v in row}) for row in L])
    print(f"\nN={p}: rectas distintas por solucion, segun el nivel")
    print("  nivel | soluciones | rectas (media, min-max) | pendientes |a| distintas (media)")
    for lv in range(level.max() + 1):
        m = level == lv
        print(f"  {lv:5d} | {m.sum():10d} | {nl[m].mean():5.2f}  {nl[m].min()}-{nl[m].max()} | {nsl[m].mean():.2f}")
    ok = level >= 0
    print("\n  soluciones por n de rectas: " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(nl[ok].tolist()).items())))
    print("  soluciones por n de pendientes |a|: " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(nsl[ok].tolist()).items())))
