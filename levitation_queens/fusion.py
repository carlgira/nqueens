"""
¿Una solucion es FUSION de otras dos (cualesquiera, no solo toroidales)?

    T(i) = A(i)  si la fila i esta abierta,   B(i)  si esta cerrada,     A != T != B

Puerta de una onda: abre las filas cuya fase (u*i mod N) cae en un arco de anchura w.
Para cada puerta G basta agrupar las soluciones por sus reinas en G y por sus reinas en el
complemento: T es fusion bajo G  <=>  otra solucion comparte sus reinas de G  y  otra distinta
comparte las del complemento.

Control: puertas al azar (subconjuntos de filas arbitrarios) de los mismos tamanos.

Uso: python fusion.py 11 13
"""
import json, os, sys, random
from collections import Counter, defaultdict
import numpy as np
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))


def wave_gates(N):
    out = {}
    for u in range(1, N // 2 + 1):
        ph = [(u * i) % N for i in range(N)]
        for x in range(N):
            for w in range(2, N - 1):                 # al menos 2 filas a cada lado
                G = frozenset(i for i in range(N) if (ph[i] - x) % N < w)
                key = min(G, frozenset(range(N)) - G, key=lambda s: (len(s), sorted(s)))
                out.setdefault(key, u)
    return out


def analyse(S, gates, N, rng):
    """-> por solucion: lista de (puerta, n de A posibles, n de B posibles)"""
    W = rng.integers(1, 2 ** 40, size=N, dtype=np.int64)
    X = (S.astype(np.int64) + 1) * W                  # contribucion de cada fila al hash
    res = defaultdict(list)
    for G in gates:
        m = np.zeros(N, bool); m[list(G)] = True
        hg = X[:, m].sum(1); hc = X[:, ~m].sum(1)
        _, ig, cg = np.unique(hg, return_inverse=True, return_counts=True)
        _, ic, cc = np.unique(hc, return_inverse=True, return_counts=True)
        na, nb = cg[ig] - 1, cc[ic] - 1
        for t in np.where((na > 0) & (nb > 0))[0]:
            res[int(t)].append((G, int(na[t]), int(nb[t])))
    return res


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    pyr = random.Random(0)
    for N in [int(x) for x in sys.argv[1:]] or [11, 13]:
        sols = mw.solutions(N)
        S = np.array(sols, dtype=np.int8)
        gates = wave_gates(N)
        res = analyse(S, list(gates), N, rng)
        # control: mismos tamanos, filas al azar
        sizes = [len(G) for G in gates]
        rg = [frozenset(pyr.sample(range(N), k)) for k in sizes]
        resr = analyse(S, rg, N, rng)
        K = {}
        f = os.path.join(HERE, 'data', f'waves_N{N}_nodeg.json')
        if os.path.exists(f):
            K = {tuple(r['solucion']): r['ondas_min'] for r in json.load(open(f))['soluciones']}
        n = len(sols)
        print(f"N={N}: {n} soluciones, {len(gates)} puertas de onda")
        print(f"   son fusion de otras dos con puerta de onda: {len(res)} ({100*len(res)/n:.1f}%)   | con puertas al azar: {len(resr)} ({100*len(resr)/n:.1f}%)")
        byu = Counter()
        bal = Counter()
        for t, L in res.items():
            byu[min(gates[G] for G, _, _ in L)] += 1
            bal[max(len(G) for G, _, _ in L)] += 1
        print(f"   por frecuencia de la puerta (la menor que sirve): {dict(sorted(byu.items()))}")
        print(f"   filas del lado pequeno de la puerta (el mayor que sirve): {dict(sorted(bal.items()))}")
        if K:
            wf = Counter(K[sols[t]] for t in res)
            wa = Counter(K.values())
            print("   por ondas minimas: " + "  ".join(f"{k}: {wf[k]}/{wa[k]} ({100*wf[k]/wa[k]:.0f}%)" for k in sorted(wa)))
        json.dump({str(t): [[sorted(G), a, b] for G, a, b in L[:5]] for t, L in res.items()},
                  open(os.path.join(HERE, 'data', f'fusion_N{N}.json'), 'w'))


# ---------------------------------------------------------------- cierre por fusion
def closure(S, gates, N, seed_mask, rng, max_rounds=30, verbose=True):
    """Partiendo de las soluciones marcadas en seed_mask, anadir toda solucion que sea fusion
    (con alguna puerta) de dos ya obtenidas. Devuelve la ronda en que aparece cada una (-1 = nunca)."""
    W = rng.integers(1, 2 ** 40, size=N, dtype=np.int64)
    X = (S.astype(np.int64) + 1) * W
    HG, HC = [], []
    for G in gates:
        m = np.zeros(N, bool); m[list(G)] = True
        HG.append(X[:, m].sum(1)); HC.append(X[:, ~m].sum(1))
    have = seed_mask.copy()
    rnd = np.where(have, 0, -1)
    for r in range(1, max_rounds + 1):
        new = np.zeros(len(S), bool)
        for hg, hc in zip(HG, HC):
            a = np.isin(hg, hg[have]) & np.isin(hc, hc[have])
            new |= a
        new &= ~have
        if not new.any():
            break
        rnd[new] = r
        have |= new
        if verbose:
            print(f"      ronda {r}: +{int(new.sum())} -> {int(have.sum())} de {len(S)}", flush=True)
    return rnd
