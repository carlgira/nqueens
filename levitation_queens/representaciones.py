"""
¿De cuantas formas distintas se puede levitar una solucion con su numero MINIMO de ondas?

Para cada clase de solucion (minimo K conocido, exacto) se prueban TODOS los subconjuntos
de K ondas no degeneradas (LP: solo amplitudes, fases y umbral) y se cuentan los que
funcionan. Tambien el margen maximo de cada uno (robustez).

  rigida      : 1 sola representacion minima (salvo el par +-k)
  degenerada  : muchas

Uso: python representaciones.py 7 8 9
Salida: data/representaciones_N{N}.json
"""
import itertools, json, os, sys, time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from scipy.optimize import linprog
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))


def margin(perm, N, X_all, cols_of, sub):
    idx = [c for n in sub for c in cols_of[n]]
    X = X_all[:, idx]
    y = mw.signs(perm, N)
    P = X.shape[1]
    A = np.hstack([-(y[:, None] * X), y[:, None], np.ones((N * N, 1))])
    r = linprog(np.r_[np.zeros(P + 1), -1.0], A_ub=A, b_ub=np.zeros(N * N),
                bounds=[(-1, 1)] * P + [(None, None), (0, None)], method='highs')
    return r.x[-1] if r.status == 0 else 0.0


def _count(args):
    cid, perm, N, K = args
    ks = mw.freq_pairs(N, False, True)
    X, owner, _ = mw.design(N, ks)
    cols_of = {n: list(np.where(owner == n)[0]) for n in range(len(ks))}
    good = []
    for sub in itertools.combinations(range(len(ks)), K):
        m = margin(perm, N, X, cols_of, sub)
        if m >= 1e-3:
            good.append(([list(ks[n]) for n in sub], round(float(m), 4)))
    return cid, good


def main(Ns):
    for N in Ns:
        t0 = time.time()
        recs = json.load(open(os.path.join(HERE, 'data', f'waves_N{N}.json')))['soluciones']
        reps = {}
        for r in recs:
            if r['exacto']:
                reps.setdefault(r['clase'], r)
        jobs = [(c, tuple(r['solucion']), N, r['ondas_min']) for c, r in reps.items()]
        with Pool(10) as p:
            res = dict(p.map(_count, jobs, chunksize=1))
        out = []
        print(f"N={N} ({time.time()-t0:.0f}s)   K = ondas minimas, R = n de representaciones minimas, margen = mejor / mediana")
        byK = {}
        for c, r in sorted(reps.items()):
            good = res[c]
            size = sum(1 for x in recs if x['clase'] == c)
            tor = len({(i + v) % N for i, v in enumerate(r['solucion'])}) == N and len({(i - v) % N for i, v in enumerate(r['solucion'])}) == N
            ms = sorted((m for _, m in good), reverse=True)
            out.append({'clase': c, 'representante': r['solucion'], 'tam_clase': size, 'toroidal': tor,
                        'K': r['ondas_min'], 'R': len(good), 'margenes': ms, 'conjuntos': [g for g, _ in good]})
            byK.setdefault(r['ondas_min'], []).append(out[-1])
        for K in sorted(byK):
            for o in sorted(byK[K], key=lambda o: o['R']):
                print(f"   K={K}  R={o['R']:5d}  margen {o['margenes'][0]:.3f} / {np.median(o['margenes']):.3f}"
                      f"  {'toroidal ' if o['toroidal'] else ''}clase {o['clase']} ({o['tam_clase']} sol) {o['representante']}")
        json.dump(out, open(os.path.join(HERE, 'data', f'representaciones_N{N}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or [7, 8, 9])
