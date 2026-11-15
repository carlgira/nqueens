"""
Orden constructivo por NIVELES de fusion.

  nivel 0   = soluciones construidas por reglas (semilla)
  nivel k+1 = soluciones que son fusion (con una puerta de onda sobre las filas) de dos
              soluciones de niveles <= k, mas sus giros y espejos (las 8 simetrias son gratis)

Semillas:
  tramo   : solo las toroidales de un tramo
  reglas  : tramo + escaleras + reflejos + intercambios a distancia + rotaciones

Para cada nivel: cuantas soluciones, y su complejidad media (cortes; ondas si hay datos).

Uso: python niveles.py 11 13 [--semilla reglas]
Salida: data/niveles_N{N}_{semilla}.npy  (nivel de cada solucion, -1 = no alcanzada)
"""
import argparse, json, os
from collections import Counter, defaultdict
import numpy as np
import min_waves as mw
from fusion import wave_gates

HERE = os.path.dirname(os.path.abspath(__file__))


def cuts(s, N):
    a = [(s[i + 1] - s[i]) % N for i in range(N - 1)]
    return N - 1 - max(Counter(a).values())


def seed_set(N, kind):
    sols = mw.solutions(N)
    S = {s for s in sols if len({(s[i + 1] - s[i]) % N for i in range(N - 1)}) == 1}
    if kind == 'reglas':
        from reflejos import all_by_rule
        from desplazamientos import build, build3
        from escaleras import family, family2
        S |= set(family(N)) | set(family2(N)) | set(all_by_rule(N)) | set(build(N)) | set(build3(N, True))
    return sols, S


def levels(sols, seed, N, max_rounds=300):
    idx = {s: i for i, s in enumerate(sols)}
    S = np.array(sols, dtype=np.int64)
    n = len(sols)
    G8 = mw.board_maps(N)
    sym = np.array([[idx[mw.apply_map(g, s)] for g in G8] for s in sols])      # (n, 8)
    rng = np.random.default_rng(0)
    W = rng.integers(1, 2 ** 40, size=N, dtype=np.int64)
    X = (S + 1) * W
    HG, HC = [], []
    for G in wave_gates(N):
        m = np.zeros(N, bool); m[list(G)] = True
        HG.append(X[:, m].sum(1)); HC.append(X[:, ~m].sum(1))
    have = np.zeros(n, bool)
    have[[idx[s] for s in seed if s in idx]] = True
    have[sym[have].ravel()] = True
    lev = np.where(have, 0, -1)
    for r in range(1, max_rounds + 1):
        new = np.zeros(n, bool)
        for hg, hc in zip(HG, HC):
            new |= np.isin(hg, hg[have]) & np.isin(hc, hc[have])
        new &= ~have
        if not new.any():
            break
        new[sym[new].ravel()] = True
        new &= ~have
        lev[new] = r
        have |= new
    return lev


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--semilla', default='reglas', choices=['tramo', 'reglas'])
    a = ap.parse_args()
    for N in a.Ns:
        sols, seed = seed_set(N, a.semilla)
        lev = levels(sols, seed, N)
        np.save(os.path.join(HERE, 'data', f'niveles_N{N}_{a.semilla}.npy'), lev)
        K = {}
        f = os.path.join(HERE, 'data', f'waves_N{N}_nodeg.json')
        if os.path.exists(f):
            K = {tuple(r['solucion']): r['ondas_min'] for r in json.load(open(f))['soluciones']}
        C = np.array([cuts(s, N) for s in sols])
        n = len(sols)
        reached = int((lev >= 0).sum())
        print(f"N={N}, semilla '{a.semilla}': nivel 0 = {int((lev == 0).sum())} | alcanzadas {reached} de {n} ({100*reached/n:.1f}%) en {lev.max()} niveles")
        print(f"   {'nivel':>5} {'soluciones':>10} {'acumulado':>10} {'cortes (media)':>15}" + ("  ondas (media)" if K else ''))
        acc = 0
        Kv = np.array([K[s] for s in sols]) if K else None
        for L in list(range(lev.max() + 1)) + [-1]:
            m = lev == L
            if not m.any():
                continue
            acc += int(m.sum()) if L >= 0 else 0
            lab = 'nunca' if L < 0 else str(L)
            print(f"   {lab:>5} {int(m.sum()):10d} {acc if L >= 0 else '':>10} {C[m].mean():15.2f}" + (f"  {Kv[m].mean():.2f}" if K else ''))
