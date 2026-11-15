"""
Censo de HACES n+m: toroidales cubiertas por n rectas paralelas de pendiente a y m de pendiente b.

No supone ninguna regla de reparto: para cada pareja de pendientes y cada colocacion de las rectas
busca todas las toroidales que eligen, en cada fila, la reina de una de las n+m rectas.
Simetrias usadas para no repetir trabajo: traslacion (una recta de cada pendiente por el origen),
escala (la segunda paralela a distancia 1) y los 8 giros (una pareja de pendientes por orbita).
Para cada solucion calcula ademas su clase real (minimo de rectas de cualquier pendiente).

Uso: ../venv/bin/python haces.py p n m [n m ...]      -> data/haces_p{p}.txt (anexa)
"""
import sys, time
from collections import Counter, defaultdict
from itertools import combinations
from orbitas import mats, apply, slope_action, inv
from tipos import nombre


def canon(T, p):
    best = None
    for s in range(p):
        t0 = T[s]
        U = tuple((T[(s + r) % p] - t0) % p for r in range(p))
        if best is None or U < best: best = U
    return best


def busca(p, lineas):
    """todas las toroidales con T(r) en {a*r + c : (a, c) en lineas} para cada fila r."""
    cand = [sorted({(a * r + c) % p for a, c in lineas}) for r in range(p)]
    T = [0] * p; out = []
    def rec(r, cols, sm, df):
        if r == p:
            out.append(tuple(T)); return
        for c in cand[r]:
            b1 = 1 << c; b2 = 1 << ((c + r) % p); b3 = 1 << ((c - r) % p)
            if cols & b1 or sm & b2 or df & b3: continue
            T[r] = c
            rec(r + 1, cols | b1, sm | b2, df | b3)
    rec(0, 0, 0, 0)
    return out


def clase(T, p, kmax):
    """minimo de rectas (pendiente 2..p-2) que cubren T, o kmax+1 si hacen falta mas."""
    def cubre(rem, k):
        if not rem: return True
        if k == 0: return False
        need = -(-len(rem) // k)
        r0 = rem[0]
        for a in range(2, p - 1):                      # la recta que cubre r0
            c = (T[r0] - a * r0) % p
            resto = [r for r in rem if (T[r] - a * r) % p != c]
            if len(rem) - len(resto) >= need or k == 1:
                if len(rem) - len(resto) >= need and cubre(resto, k - 1): return True
        # r0 puede estar en una recta pequeña: probar tambien sin la poda
        for a in range(2, p - 1):
            c = (T[r0] - a * r0) % p
            resto = [r for r in rem if (T[r] - a * r) % p != c]
            if len(rem) - len(resto) < need and cubre(resto, k - 1): return True
        return False
    for k in range(1, kmax + 1):
        if cubre(list(range(p)), k): return k
    return kmax + 1


def parejas(p, simetrica):
    """un representante (a, b) por orbita de los 8 giros (y del intercambio si n == m)."""
    D = mats(p)[::p - 1]
    vistos, reps = set(), []
    for a in range(2, p - 1):
        for b in range(2, p - 1):
            if a == b or (a, b) in vistos: continue
            orb = set()
            for M in D:
                x, y = slope_action(M, a, p), slope_action(M, b, p)
                orb.add((x, y))
                if simetrica: orb.add((y, x))
            vistos |= orb; reps.append((a, b, len(orb)))
    return reps, D


def censo(p, n, m):
    reps, D = parejas(p, n == m)
    res = {}                                            # T canonica -> pareja representante
    t0 = time.time()
    for a, b, tam in reps:
        if n >= 2:
            confA = [(0, 1) + c for c in combinations(range(2, p), n - 2)]
            confB = [(0,) + d for d in combinations(range(1, p), m - 1)]
        else:
            confA = [(0,)]
            confB = [(0, 1) + d for d in combinations(range(2, p), m - 2)]
        base = set()
        for cs in confA:
            for ds in confB:
                lin = [(a, c) for c in cs] + [(b, d) for d in ds]
                for T in busca(p, lin):
                    # todas las rectas del haz tienen que hacer falta
                    quien = [[i for i, (s, c) in enumerate(lin) if (s * r + c) % p == T[r]] for r in range(p)]
                    solas = {q[0] for q in quien if len(q) == 1}
                    if len(solas) < n + m: continue
                    base.add(T)
        for T in base:
            for u in range(1, p):
                U = [0] * p
                for r in range(p): U[u * r % p] = u * T[r] % p
                for M in D:
                    V = canon([int(x) for x in apply(M, U, p)], p)
                    res.setdefault(V, (a, b))
    cl = Counter()
    for T, (a, b) in res.items():
        cl[(clase(T, p, n + m), f"{nombre(a, p)},{nombre(b, p)}")] += 1
    return len(res), cl, time.time() - t0


if __name__ == "__main__":
    p = int(sys.argv[1]); nm = list(map(int, sys.argv[2:]))
    with open(f"data/haces_p{p}.txt", "a") as f:
        for n, m in zip(nm[::2], nm[1::2]):
            tot, cl, dt = censo(p, n, m)
            linea = (f"p={p} haz {n}+{m}: {tot} orbitas de traslacion ({dt:.0f} s) | "
                     + " · ".join(f"clase {k} [{t}]: {v}" for (k, t), v in sorted(cl.items())))
            print(linea, flush=True); f.write(linea + "\n")
