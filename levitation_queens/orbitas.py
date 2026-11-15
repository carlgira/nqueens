"""
Orbitas de las toroidales bajo TODO el grupo de simetrias del toro.
Grupo: traslaciones (p^2) x multiplicar el tablero por u (p-1) x 8 "giros" que permutan las
cuatro direcciones prohibidas (filas, columnas, dos diagonales). Total 8(p-1)p^2.
Los 8 giros incluyen (r,c)->(r+c, c-r), que en el toro es un giro de 45 grados.
Sobre la pendiente a de una recta actuan asi: a -> -a, 1/a, (a-1)/(a+1) y combinaciones.
Uso: python orbitas.py 13 17 19   -> data/orbitas_p{p}.json
"""
import sys, json, itertools
from collections import Counter, defaultdict
import numpy as np


def inv(a, p): return pow(a % p, p - 2, p)

def mats(p):
    """las 8(p-1) matrices: u * D, D en el grupo diedrico generado por S=(c,r), R=(r+c, c-r)."""
    D = set()
    base = [((1, 0), (0, 1))]
    gens = [((0, 1), (1, 0)), ((1, 1), (p - 1, 1))]
    seen = set()
    def proj(M):
        (a, b), (c, d) = M
        for x in (a, b, c, d):
            if x % p:
                k = inv(x, p); break
        return tuple(tuple(y * k % p for y in row) for row in M)
    def mul(A, B):
        return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(2)) % p for j in range(2)) for i in range(2))
    todo = [proj(base[0])]; seen = {todo[0]}
    while todo:
        M = todo.pop()
        for g in gens:
            N = proj(mul(g, M))
            if N not in seen: seen.add(N); todo.append(N)
    assert len(seen) == 8, len(seen)
    out = []
    for M in sorted(seen):
        for u in range(1, p):
            out.append(tuple(tuple(u * y % p for y in row) for row in M))
    return out

def apply(M, T, p):
    (a, b), (c, d) = M
    r = np.arange(p); T = np.asarray(T)
    R = (a * r + b * T) % p; C = (c * r + d * T) % p
    out = np.empty(p, dtype=np.int64); out[R] = C
    return out

def canon_tr(T, p):
    """representante de la orbita de traslacion: T(0)=0 y minimo lexicografico."""
    T = np.asarray(T); best = None
    for s in range(p):
        U = tuple(int(x) for x in (np.roll(T, -s) - T[s]) % p)
        if best is None or U < best: best = U
    return best

def slope_action(M, a, p):
    """pendiente de la imagen de una recta de pendiente a (direccion (1,a))."""
    (m11, m12), (m21, m22) = M
    x = (m11 + m12 * a) % p; y = (m21 + m22 * a) % p
    return y * inv(x, p) % p

def lines_of(T, p):
    """rectas (a, c, mascara de filas) con >= 1 reina, pendiente 2..p-2."""
    L = []
    for a in range(2, p - 1):
        by = defaultdict(int)
        for r in range(p): by[(T[r] - a * r) % p] |= 1 << r
        for c, m in by.items(): L.append((a, c, m))
    return L

def min_covers(T, p, kmax=None):
    """todas las coberturas minimas por rectas (cada reina en al menos una recta)."""
    L = lines_of(T, p); full = (1 << p) - 1
    thr = [[j for j, (a, c, m) in enumerate(L) if m >> r & 1] for r in range(p)]
    maxsz = max(bin(m).count('1') for _, _, m in L)
    for k in range(1, p + 1):
        res = set()
        def dfs(cov, chosen):
            if cov == full:
                res.add(tuple(sorted(chosen))); return
            left = bin(full & ~cov).count('1')
            if len(chosen) + (left + maxsz - 1) // maxsz > k: return
            r = (~cov & full & -(~cov & full)).bit_length() - 1
            for j in thr[r]:
                dfs(cov | L[j][2], chosen + [j])
        dfs(0, [])
        if res: return k, [[L[j] for j in cv] for cv in res]

def load(p):
    """orbitas de traslacion con su clase, leyendo el volcado de minrectas."""
    reps = {}
    with open(f"data/clases_toro_p{p}.txt") as f:
        for ln in f:
            v = ln.split()
            if v[1] != '0': continue
            T = [int(x) for x in v[1:]]
            reps[canon_tr(T, p)] = int(v[0])
    return reps

def orbits(p, reps=None):
    reps = reps or load(p); G = mats(p)
    seen = set(); out = []
    for T in sorted(reps):
        if T in seen: continue
        orb = set(); stab = 0
        for M in G:
            U = canon_tr(apply(M, T, p), p)
            assert U in reps and reps[U] == reps[T]
            orb.add(U); stab += U == T
        seen |= orb
        out.append(dict(rep=T, clase=reps[T], tam=len(orb), stab=stab))
        assert len(orb) * stab == len(G)
    return out, reps


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]]:
        orbs, reps = orbits(p)
        print(f"p={p}: {len(reps)} orbitas de traslacion, {len(orbs)} orbitas del grupo completo (orden 8(p-1)={8*(p-1)})")
        bycl = defaultdict(list)
        for o in orbs: bycl[o['clase']].append(o)
        for cl in sorted(bycl):
            sizes = sorted(o['tam'] for o in bycl[cl])
            print(f"  clase {cl}: {sum(sizes)} por orbita de traslacion = {' + '.join(map(str, sizes))}")
        json.dump([dict(rep=list(o['rep']), clase=o['clase'], tam=o['tam'], stab=o['stab']) for o in orbs], open(f"data/orbitas_p{p}.json", "w"))
