"""Construye el grafo completo dirigido de soluciones de N-reinas y lo guarda en .npz.

Nodo   : solución q (q[i] = columna de la reina en la fila i), es decir una matriz de permutación P.
Arista : a -> b con T = P_b · P_a^T (permutación). Como permutación: t[q_a[i]] = q_b[i].
Se guardan atributos precalculados por arista para que los criterios sean máscaras baratas.

Uso: python build_graph.py N      -> graph_N.npz
"""
import sys
import numpy as np


def solve(n):
    sols, q = [], [0] * n
    cols, d1, d2 = set(), set(), set()

    def rec(r):
        if r == n:
            sols.append(tuple(q))
            return
        for c in range(n):
            if c in cols or (r - c) in d1 or (r + c) in d2:
                continue
            q[r] = c
            cols.add(c); d1.add(r - c); d2.add(r + c)
            rec(r + 1)
            cols.discard(c); d1.discard(r - c); d2.discard(r + c)

    rec(0)
    return np.array(sols, dtype=np.int8)


def is_toroidal(S):
    """Solución toroidal: q(i)+i y q(i)-i son todos distintos módulo N (diagonales rotas incluidas).
    S: array (m, n). Devuelve máscara booleana (m,)."""
    S = np.asarray(S, dtype=np.int64)
    n = S.shape[1]
    ar = np.arange(n)
    ok1 = (np.sort((S + ar) % n, axis=1) == ar).all(axis=1)
    ok2 = (np.sort((S - ar) % n, axis=1) == ar).all(axis=1)
    return ok1 & ok2


def sym_images(q):
    """Las 8 simetrías de D4 aplicadas a una solución (como permutación fila->columna)."""
    n = len(q)
    p = np.asarray(q)
    inv = np.empty(n, dtype=p.dtype); inv[p] = np.arange(n)       # transposición
    imgs = []
    for base in (p, inv):
        imgs.append(base)                    # id / transpuesta
        imgs.append(base[::-1])              # reflejo vertical
        imgs.append((n - 1) - base)          # reflejo horizontal
        imgs.append((n - 1) - base[::-1])    # rotación 180
    return imgs


def cycle_type(perm):
    n = len(perm)
    seen = [False] * n
    lens = []
    for i in range(n):
        if not seen[i]:
            l, j = 0, i
            while not seen[j]:
                seen[j] = True; j = perm[j]; l += 1
            lens.append(l)
    return tuple(sorted(lens, reverse=True)), len(lens)


def main(n):
    S = solve(n).astype(np.int64)
    m = len(S)
    index = {tuple(s): i for i, s in enumerate(S)}

    # Orbita D4 de cada solución (id de la solución en la que cae cada imagen)
    sym = np.full((m, 8), -1, dtype=np.int32)
    for i, s in enumerate(S):
        for k, im in enumerate(sym_images(s)):
            sym[i, k] = index[tuple(im)]

    src, dst, ham, cay, ctype_id, ncyc, fixed = [], [], [], [], [], [], []
    ctype_ids = {}
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            t = np.empty(n, dtype=np.int64)
            t[S[a]] = S[b]                       # T = P_b P_a^T como permutación
            ct, nc = cycle_type(t)
            src.append(a); dst.append(b)
            ham.append(int((S[a] != S[b]).sum()))
            cay.append(n - nc)                   # distancia de Cayley = n - nº ciclos
            ctype_id.append(ctype_ids.setdefault(ct, len(ctype_ids)))
            ncyc.append(nc)
            fixed.append(int((t == np.arange(n)).sum()))

    types = sorted(ctype_ids, key=ctype_ids.get)
    np.savez_compressed(
        f"graph_{n}.npz", n=n, sols=S, sym=sym,
        src=np.array(src, np.int32), dst=np.array(dst, np.int32),
        hamming=np.array(ham, np.int16), cayley=np.array(cay, np.int16),
        ctype=np.array(ctype_id, np.int32), ncyc=np.array(ncyc, np.int16),
        fixed=np.array(fixed, np.int16),
        ctype_names=np.array([str(t) for t in types]),
    )
    print(f"N={n}: {m} soluciones, {len(src)} aristas dirigidas, {len(types)} tipos de ciclos distintos")


if __name__ == "__main__":
    main(int(sys.argv[1]))
