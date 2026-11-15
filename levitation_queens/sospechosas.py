"""
Camino inverso: filtrar por aritmetica y solo despues buscar ondas.

1. Enumerar las soluciones de N y agruparlas por simetria.
2. Sospechosa = clase con pocos pasos distintos: D = n de valores de sigma(i+1)-sigma(i)
   mod N (minimo sobre las 8 simetrias) <= umbral. Se descartan las toroidales (D = 1).
3. Solo para las sospechosas: MILP de ondas minimas (vocabulario no degenerado).

Uso: python sospechosas.py 12 13 [--umbral 3] [--time-limit 120]
Salida: data/sospechosas_N{N}.json  (reanudable: data/checkpoints/sospechosas_N{N}.jsonl)
"""
import argparse, json, os, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))


def feats(s, N):
    st = [(s[i + 1] - s[i]) % N for i in range(N - 1)]
    return len(set(st)), 1 + sum(st[i] != st[i + 1] for i in range(N - 2))


def _solve(args):
    cid, rep, N, tl = args
    ks = mw.freq_pairs(N, False, True)
    X, o, _ = mw.design(N, ks)
    t0 = time.time()
    sup, ex, lo = mw.min_support(rep, N, ks, X, o, 1e-3, tl)
    waves, t, m = mw.best_params(rep, N, [ks[i] for i in sup])
    assert mw.verify(rep, N, waves, t)
    return cid, len(sup), bool(ex), lo, waves, round(m, 4), round(time.time() - t0, 1)


def run(N, thr, tl, workers):
    t0 = time.time()
    sols = mw.solutions(N)
    reps = mw.classes(sols, N)
    info = []
    for cid, (rep, members) in enumerate(reps.items()):
        D, runs, best = min((feats(s, N) + (s,)) for s, _ in members)
        info.append((cid, D, runs, best, len(members)))
    hist = Counter(D for _, D, _, _, _ in info)
    sus = [x for x in info if 1 < x[1] <= thr]
    print(f"N={N}: {len(sols)} soluciones, {len(reps)} clases ({time.time()-t0:.0f}s) | clases por pasos distintos "
          f"{dict(sorted(hist.items()))}", flush=True)
    print(f"   sospechosas (2 <= D <= {thr}): {len(sus)} clases = {100*len(sus)/len(reps):.1f}% del total", flush=True)
    ck = os.path.join(HERE, 'data', 'checkpoints', f'sospechosas_N{N}.jsonl')
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    done = {}
    if os.path.exists(ck):
        for line in open(ck):
            try:
                d = json.loads(line); done[d['clase']] = d
            except json.JSONDecodeError:
                pass
    jobs = [(cid, best, N, tl) for cid, D, runs, best, n in sus if cid not in done]
    meta = {cid: (D, runs, best, n) for cid, D, runs, best, n in sus}
    if jobs:
        with Pool(workers) as p, open(ck, 'a') as fh:
            for cid, K, ex, lo, waves, m, secs in p.imap_unordered(_solve, jobs, chunksize=1):
                D, runs, best, n = meta[cid]
                d = {'clase': cid, 'solucion': list(best), 'tam_clase': n, 'pasos_distintos': D, 'tramos': runs,
                     'ondas': K, 'exacto': ex, 'cota_inferior': lo, 'margen': m, 'ondas_param': waves, 'seg': secs}
                fh.write(json.dumps(d) + '\n'); fh.flush(); os.fsync(fh.fileno())
                done[cid] = d
                print(f"   [{sum(c in done for c in meta)}/{len(meta)}] D={D} tramos={runs} -> {K} ondas"
                      f"{'' if ex else ' (cota)'}  {list(best)}  {secs:.0f}s", flush=True)
    rows = [done[c] for c in meta]
    json.dump({'N': N, 'umbral': thr, 'n_clases': len(reps), 'hist_pasos': dict(hist), 'sospechosas': rows},
              open(os.path.join(HERE, 'data', f'sospechosas_N{N}.json'), 'w'), indent=1)
    tab = defaultdict(Counter)
    for d in rows:
        tab[d['pasos_distintos']][(d['ondas'], d['exacto'])] += 1
    print(f"N={N}: resultado de las {len(rows)} sospechosas (pasos distintos -> ondas: clases)")
    for D in sorted(tab):
        print(f"   D={D}: " + ", ".join(f"{k} ondas{'' if e else ' (cota)'}: {v}" for (k, e), v in sorted(tab[D].items())), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--umbral', type=int, default=3)
    ap.add_argument('--time-limit', type=float, default=120)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    for N in a.Ns:
        run(N, a.umbral, a.time_limit, a.workers)
