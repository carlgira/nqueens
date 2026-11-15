"""
Transferencia: reutilizar los conjuntos de ondas encontrados en N = 4..10 para
levitar soluciones de un N mayor, sin MILP.

Para cada conjunto de vectores k (de data/waves_N*.json) y cada solucion del N
nuevo, un LP decide si reajustando SOLO amplitudes, fases y umbral las N
casillas mas brillantes son las reinas. Dos formas de trasladar k:

  mismo_k      k identico (mismos ciclos por tablero)
  misma_lambda k * N_nuevo / N_origen (misma longitud de onda en casillas; k no entero)

Se prueban las 8 orientaciones de cada conjunto (k -> L k). Los conjuntos se
recorren de menos a mas ondas y cada solucion se queda con el primero que sirve,
asi que el resultado es una COTA SUPERIOR del minimo de ondas.

Uso: python transfer.py 11 12
Salida: data/transfer_N{N}.json
"""
import argparse, json, math, os, time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from scipy.optimize import linprog
import min_waves as mw

LS = [(1, 0, 0, 1), (0, 1, -1, 0), (-1, 0, 0, -1), (0, -1, 1, 0),
      (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0)]
DELTA = 1e-3


def canon(ks):
    best = None
    for a, b, c, d in LS:
        t = []
        for u, v in ks:
            x = (round(a * u + b * v, 9), round(c * u + d * v, 9))
            t.append(max(x, (-x[0], -x[1])))
        t = tuple(sorted(t))
        if best is None or t < best:
            best = t
    return best


def load_supports(Nnew, Ns=range(4, 11)):
    """-> lista de (esquema, n_ondas, ks, origen) sin duplicados por simetria."""
    out, seen = [], set()
    for N in Ns:
        for r in json.load(open(os.path.join(mw.HERE, 'data', f'waves_N{N}.json')))['soluciones']:
            ks = [tuple(w['k']) for w in r['ondas']]
            for esquema, kk in (('mismo_k', ks),
                                ('misma_lambda', [(u * Nnew / N, v * Nnew / N) for u, v in ks])):
                key = (esquema, canon(kk))
                if key in seen:
                    continue
                seen.add(key)
                out.append((esquema, len(kk), list(key[1]), {'N': N, 'id': r['id']}))
    out.sort(key=lambda s: (s[1], s[0] != 'mismo_k'))
    return out


def lp_margin(perm, N, ks):
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    I, J = I.ravel(), J.ravel()
    cols = []
    for u, v in ks:
        th = 2 * np.pi * (u * I + v * J) / N
        cols += [np.cos(th), np.sin(th)]
    X = np.array(cols).T
    y = mw.signs(perm, N)
    P = X.shape[1]
    A = np.hstack([-(y[:, None] * X), y[:, None], np.ones((N * N, 1))])
    r = linprog(np.r_[np.zeros(P + 1), -1.0], A_ub=A, b_ub=np.zeros(N * N),
                bounds=[(-1, 1)] * P + [(None, None), (0, None)], method='highs')
    if r.status != 0:
        return 0.0, None
    return r.x[-1], r.x


def params(N, ks, x):
    waves = []
    for n, (u, v) in enumerate(ks):
        a, b = x[2 * n], x[2 * n + 1]
        A = math.hypot(a, b)
        if A < 1e-9:
            continue
        kn = math.hypot(u, v)
        waves.append({'k': [round(u, 6), round(v, 6)], 'amplitud': round(A, 6),
                      'fase_rad': round(math.atan2(-b, a), 6),
                      'longitud_onda_casillas': round(N / kn, 6),
                      'direccion_grados': round(math.degrees(math.atan2(u, v)), 3)})
    return waves, float(x[-2])


SUPS = None


def _init(sups):
    global SUPS
    SUPS = sups


def _one(args):
    perm, N = args
    for esquema, K, ks, orig in SUPS:
        for a, b, c, d in LS:
            kk = [(a * u + b * v, c * u + d * v) for u, v in ks]
            m, x = lp_margin(perm, N, kk)
            if m >= DELTA:
                waves, t = params(N, kk, x)
                if mw.verify(perm, N, waves, t):
                    return {'ondas': len(waves), 'esquema': esquema, 'origen': orig,
                            'margen': round(m, 6), 'umbral': round(t, 6), 'ondas_param': waves}
    return None


def run(N, workers):
    t0 = time.time()
    sups = load_supports(N)
    sols = mw.solutions(N)
    reps = mw.classes(sols, N)
    with Pool(workers, initializer=_init, initargs=(sups,)) as pool:
        res = []
        for n, r in enumerate(pool.imap(_one, [(rep, N) for rep in reps], chunksize=1), 1):
            res.append(r)
            if n % 100 == 0:
                print(f'  {n}/{len(reps)} clases, {sum(x is not None for x in res)} levitadas, {time.time()-t0:.0f}s', flush=True)
    idx = {s: n for n, s in enumerate(sols)}
    recs = [None] * len(sols)
    for cid, ((rep, members), r) in enumerate(zip(reps.items(), res)):
        for sol, g in members:
            rec = {'id': idx[sol], 'solucion': list(sol), 'clase': cid}
            if r is None:
                rec.update({'ondas_cota': None})
            else:
                # re-orientar las ondas del representante a este miembro: k' = L k, y reajustar fases
                (a, b, c, d), _ = g
                kk = [(a * w['k'][0] + b * w['k'][1], c * w['k'][0] + d * w['k'][1]) for w in r['ondas_param']]
                m, x = lp_margin(sol, N, kk)
                waves, t = params(N, kk, x)
                assert m >= DELTA and mw.verify(sol, N, waves, t)
                rec.update({'ondas_cota': len(waves), 'esquema': r['esquema'], 'origen': r['origen'],
                            'margen': round(m, 6), 'umbral': round(t, 6), 'ondas': waves})
            recs[idx[sol]] = rec
    h = Counter(r['ondas_cota'] or 0 for r in recs)
    e = Counter(r.get('esquema') for r in recs if r['ondas_cota'])
    o = Counter(r['origen']['N'] for r in recs if r['ondas_cota'])
    out = {'N': N, 'delta': DELTA, 'n_soluciones': len(sols), 'n_clases': len(reps),
           'n_conjuntos_probados': len(sups),
           'nota': 'ondas_cota es COTA SUPERIOR del minimo (solo conjuntos de ondas heredados de N=4..10)',
           'soluciones': recs}
    json.dump(out, open(os.path.join(mw.HERE, 'data', f'transfer_N{N}.json'), 'w'), indent=1, ensure_ascii=False)
    print(f"N={N}: {len(sols)} sol, {len(reps)} clases, {len(sups)} conjuntos | "
          f"ondas (cota) {dict(sorted(h.items()))}  [0 = ninguno sirve] | esquema {dict(e)} | "
          f"origen N {dict(sorted(o.items()))} | {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    for N in a.Ns:
        run(N, a.workers)
