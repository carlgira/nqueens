"""
Refina las clases NO certificadas de un checkpoint de min_waves.py, preguntando
directamente "¿hay alguna con <= K-1 ondas?" (MILP con sum z <= K-1).

  - si es infactible (probado)  -> K es el minimo exacto
  - si encuentra una            -> baja a esa y se vuelve a preguntar con una menos
  - si se acaba el tiempo        -> queda como cota, con la cota inferior mejorada

Reanudable: cada resultado se AÑADE como linea nueva al mismo checkpoint (la ultima
linea de una clase manda). Al terminar, relanzar min_waves.py con los mismos
argumentos reconstruye el JSON final sin recalcular nada.

  python refinar_ckpt.py 11 --nodeg --time-limit 7200 [--workers 7]
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import min_waves as mw


def at_most(perm, N, ks, X, owner, delta, kmax, time_limit):
    """MILP: minimo sum z con sum z <= kmax. Devuelve (sup | None, estado)."""
    y = mw.signs(perm, N)
    P, K = X.shape[1], len(ks)
    A1 = np.hstack([y[:, None] * X, -y[:, None], np.zeros((N * N, K))])
    E = np.zeros((P, K)); E[np.arange(P), owner] = 1
    Ic = np.eye(P)
    cons = [LinearConstraint(A1, delta, np.inf),
            LinearConstraint(np.hstack([Ic, np.zeros((P, 1)), -E]), -np.inf, 0),
            LinearConstraint(np.hstack([-Ic, np.zeros((P, 1)), -E]), -np.inf, 0),
            LinearConstraint(np.r_[np.zeros(P + 1), np.ones(K)][None, :], -np.inf, kmax)]
    obj = np.r_[np.zeros(P + 1), np.ones(K)]
    lb = np.r_[-np.ones(P), -np.inf, np.zeros(K)]
    ub = np.r_[np.ones(P), np.inf, np.ones(K)]
    r = milp(obj, constraints=cons, bounds=Bounds(lb, ub),
             integrality=np.r_[np.zeros(P + 1), np.ones(K)],
             options={'time_limit': time_limit, 'mip_rel_gap': 0})
    if r.status == 2:
        return None, 'infactible'
    if r.x is None:
        return None, 'tiempo'
    sup = [n for n in range(K) if r.x[P + 1 + n] > 0.5]
    return sup, ('optimo' if r.status == 0 else 'tiempo')


def _refine(args):
    cid, rep, N, ks, delta, cur_k, time_limit, fisico, nodeg = args
    X, owner, _ = mw.design(N, ks)
    t0 = time.time()
    best = cur_k                      # lista de k de la mejor solucion conocida
    lower = None
    while True:
        left = time_limit - (time.time() - t0)
        if left < 5:
            break
        sup, st = at_most(rep, N, ks, X, owner, delta, len(best) - 1, left)
        if st == 'infactible':
            lower = len(best)         # probado: no hay con menos
            break
        if sup is None:
            break                     # tiempo agotado sin respuesta
        best = [list(ks[i]) for i in sup]
        if st == 'optimo':
            lower = len(best)
            break
    return cid, best, lower, time.time() - t0


def main(N, time_limit, fisico, nodeg, workers):
    tag = f'waves_N{N}' + ('_fisico' if fisico else '') + ('_nodeg' if nodeg else '')
    ckpath = os.path.join(mw.HERE, 'data', 'checkpoints', tag + '.jsonl')
    last = {}
    for line in open(ckpath):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        last[d['clase']] = d
    pend = [d for d in last.values() if not d['exacto'] and d.get('time_limit', 0) < time_limit]
    print(f"N={N}: {len(last)} clases en checkpoint, {len(pend)} sin certificar a refinar "
          f"(limite {time_limit:.0f}s cada una)", flush=True)
    if not pend:
        return
    sols = mw.solutions(N)
    reps = list(mw.classes(sols, N))
    ks = mw.freq_pairs(N, fisico, nodeg)
    jobs = [(d['clase'], reps[d['clase']], N, ks, d['delta'], d['sup_k'], time_limit, fisico, nodeg)
            for d in pend]
    with Pool(min(workers, len(jobs))) as pool, open(ckpath, 'a') as fh:
        for cid, best, lower, secs in pool.imap_unordered(_refine, jobs, chunksize=1):
            old = last[cid]
            d = dict(old)
            d.update({'sup_k': best, 'exacto': lower is not None and lower == len(best),
                      'cota_inferior': lower if lower is not None else old['cota_inferior'],
                      'seg': round(old['seg'] + secs, 1), 'time_limit': time_limit})
            if len(best) != len(old['sup_k']):
                d['libre'] = mw.one_free_wave(reps[cid], N, 8, fisico) if len(best) != 1 else 'rejilla'
            fh.write(json.dumps(d) + '\n'); fh.flush(); os.fsync(fh.fileno())
            print(f"  clase {cid}: antes {len(old['sup_k'])} (>= {old['cota_inferior']}), ahora {len(best)} "
                  f"{'EXACTO' if d['exacto'] else '(cota)'}  {secs:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--time-limit', type=float, default=7200)
    ap.add_argument('--fisico', action='store_true')
    ap.add_argument('--nodeg', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    main(a.N, a.time_limit, a.fisico, a.nodeg, a.workers)
