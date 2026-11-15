"""
Recalcula con mas tiempo las clases que quedaron sin certificar (exacto = false)
en data/waves_N{N}[_fisico].json y actualiza el fichero.

  python refinar.py 10 --time-limit 7200 [--fisico]
"""
import argparse, json, os
from multiprocessing import Pool
import min_waves as mw


def main(N, time_limit, fisico, workers):
    path = os.path.join(mw.HERE, 'data', f'waves_N{N}' + ('_fisico' if fisico else '') + '.json')
    d = json.load(open(path))
    sols = mw.solutions(N)
    reps = mw.classes(sols, N)
    reps_l = list(reps.items())
    pend = sorted({r['clase'] for r in d['soluciones'] if not r['exacto']})
    print(f'N={N}: clases pendientes {pend}', flush=True)
    if not pend:
        return
    jobs = [(reps_l[c][0], N, d['delta'], time_limit, 8, fisico) for c in pend]
    with Pool(min(workers, len(jobs))) as pool:
        res = pool.map(mw._solve_class, jobs, chunksize=1)
    ks = mw.freq_pairs(N, fisico)
    idx = {s: n for n, s in enumerate(sols)}
    for c, (sup, exact, lower, free) in zip(pend, res):
        old = d['soluciones'][idx[reps_l[c][0]]]
        print(f'  clase {c}: antes {old["ondas_min"]} (>= {old["cota_inferior"]}), '
              f'ahora {len(sup)} exacto={exact} cota={lower}', flush=True)
        for sol, g in reps_l[c][1]:
            (a, b, cc, dd), _ = g
            sup_k = [((a * ks[n][0] + b * ks[n][1]) % N, (cc * ks[n][0] + dd * ks[n][1]) % N) for n in sup]
            waves, t, m = mw.best_params(sol, N, sup_k)
            assert mw.verify(sol, N, waves, t)
            r = d['soluciones'][idx[sol]]
            r.update({'ondas_min': len(sup), 'exacto': bool(exact),
                      'cota_inferior': len(sup) if exact else max(lower or 0, r['cota_inferior'] or 0),
                      'umbral': round(t, 6), 'margen': round(m, 6), 'ondas': waves,
                      'una_onda_libre': mw._free_k_for(free, g, waves)})
    json.dump(d, open(path, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--time-limit', type=float, default=7200)
    ap.add_argument('--fisico', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    main(a.N, a.time_limit, a.fisico, a.workers)
