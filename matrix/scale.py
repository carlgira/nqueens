"""Escala los criterios principales hasta N=12 sin guardar el grafo completo.

Por cada solución a calcula en bloque T = P_b P_a^T para todas las b > a (los tres criterios son simétricos)
y aplica: hamming<=4, hamming<=5, cayley<=2, cayley<=3, involución (T^2 = id).
Reglas del grafo: sin aristas entre soluciones de la misma órbita D4 y sin aristas toroidal-toroidal.

Uso: python scale.py 8 9 10 11 12
"""
import sys, time
import numpy as np
from build_graph import solve, sym_images, is_toroidal

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def components(m, a, b):
    if HAVE_SCIPY:
        g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(m, m))
        return connected_components(g, directed=False)[1]
    p = np.arange(m)
    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            p[rx] = ry
    return np.array([find(i) for i in range(m)])


def cycle_counts(T):
    n = T.shape[1]
    ar = np.arange(n)
    lab = np.broadcast_to(ar, T.shape).copy()
    for _ in range(n):
        lab = np.minimum(lab, np.take_along_axis(lab, T, axis=1))
    return (lab == ar).sum(axis=1)


def run(n):
    t0 = time.time()
    S = solve(n).astype(np.int64); m = len(S)
    index = {tuple(s): i for i, s in enumerate(S)}
    # órbitas D4
    parent = np.arange(m)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, s in enumerate(S):
        for im in sym_images(s):
            r1, r2 = find(i), find(index[tuple(im)])
            if r1 != r2:
                parent[r1] = r2
    orbit = np.array([find(i) for i in range(m)])
    tor = is_toroidal(S)
    ar = np.arange(n)

    names = ["hamming<=4", "hamming<=5", "cayley<=2", "cayley<=3", "involución"]
    E = {k: ([], []) for k in names}
    for a in range(m - 1):
        bs = np.arange(a + 1, m)
        ok = (orbit[bs] != orbit[a]) & ~(tor[a] & tor[bs])
        bs = bs[ok]
        if len(bs) == 0:
            continue
        Sb = S[bs]
        ham = (Sb != S[a]).sum(axis=1)
        T = np.empty_like(Sb)
        T[np.arange(len(bs))[:, None], S[a][None, :]] = Sb
        cyc = cycle_counts(T)
        cay = n - cyc
        inv = (np.take_along_axis(T, T, axis=1) == ar).all(axis=1)
        for name, mask in (("hamming<=4", ham <= 4), ("hamming<=5", ham <= 5),
                           ("cayley<=2", cay <= 2), ("cayley<=3", cay <= 3), ("involución", inv)):
            sel = bs[mask]
            if len(sel):
                E[name][0].append(np.full(len(sel), a, dtype=np.int32)); E[name][1].append(sel.astype(np.int32))

    print(f"\n=== N={n}: {m} soluciones ({int(tor.sum())} toroidales), {len(set(orbit.tolist()))} órbitas D4  "
          f"({time.time()-t0:.0f}s)")
    for name in names:
        if E[name][0]:
            a = np.concatenate(E[name][0]); b = np.concatenate(E[name][1])
        else:
            a = b = np.array([], dtype=np.int32)
        lab = components(m, a, b)
        sizes = np.bincount(lab)
        order = np.argsort(-sizes)
        nontriv = int((sizes > 1).sum())
        big = sizes[order[:6]].tolist()
        # órbitas D4 distintas dentro de la mayor componente
        main = order[0]
        orb_in_main = len(set(orbit[lab == main].tolist()))
        print(f"  {name:<11} aristas={len(a):>9}  comps={len(sizes):>6}  no triviales={nontriv:>5}  "
              f"mayor={sizes[main]:>6} ({100*sizes[main]/m:5.1f}%)  aislados={int((sizes==1).sum()):>6}  "
              f"órb en mayor={orb_in_main:>4}  top={big}")
        sys.stdout.flush()


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        run(n)
