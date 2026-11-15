"""
Nucleo de una solucion: ondas que aparecen en TODAS sus representaciones minimas.

Una onda w de la representacion conocida es de nucleo  <=>  prohibiendo w no existe
ninguna representacion con K ondas (MILP de factibilidad, sum z <= K, z_w = 0).
Rigida = todas sus K ondas son de nucleo (representacion unica).

Reanudable: una linea por clase en data/checkpoints/nucleo_N{N}.jsonl.

Uso: python nucleo.py 9 10 11 [--kmax 5] [--time-limit 120] [--muestra-por-K 40]
"""
import argparse, json, os, random, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
from refinar_ckpt import at_most
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))


def load(N):
    f = f'waves_N{N}_nodeg.json'          # mismo vocabulario (no degeneradas) que el MILP de nucleo
    return json.load(open(os.path.join(HERE, 'data', f)))['soluciones']


def norm(k, N):
    k = (k[0] % N, k[1] % N)
    return min(k, ((-k[0]) % N, (-k[1]) % N))


def _core(args):
    cid, perm, N, sup, tl = args
    ks = mw.freq_pairs(N, False, True)
    kidx = {k: i for i, k in enumerate(ks)}
    K = len(sup)
    core, unknown = [], []
    t0 = time.time()
    for w in sup:
        n = kidx[norm(w, N)]
        keep = [i for i in range(len(ks)) if i != n]
        sub = [ks[i] for i in keep]
        X, o, _ = mw.design(N, sub)
        s, st = at_most(perm, N, sub, X, o, 1e-3, K, tl)
        if st == 'infactible':
            core.append(list(w))
        elif s is None:
            unknown.append(list(w))
    return cid, core, unknown, time.time() - t0


def run(N, kmax, tl, per_k, workers):
    recs = load(N)
    reps = {}
    for r in recs:
        if r['exacto'] and r['ondas_min'] and 2 <= r['ondas_min'] <= kmax:
            reps.setdefault(r['clase'], r)
    size = Counter(r['clase'] for r in recs)
    byK = defaultdict(list)
    for c, r in reps.items():
        byK[r['ondas_min']].append(c)
    rng = random.Random(N)
    chosen = []
    for K, cs in byK.items():
        chosen += cs if not per_k or len(cs) <= per_k else rng.sample(cs, per_k)
    ck = os.path.join(HERE, 'data', 'checkpoints', f'nucleo_N{N}.jsonl')
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    done = {}
    if os.path.exists(ck):
        for line in open(ck):
            try:
                d = json.loads(line); done[d['clase']] = d
            except json.JSONDecodeError:
                pass
    jobs = [(c, tuple(reps[c]['solucion']), N, [tuple(w['k']) for w in reps[c]['ondas']], tl)
            for c in chosen if c not in done]
    print(f"N={N}: {len(chosen)} clases (K=2..{kmax}), {len(done)} ya hechas, {len(jobs)} pendientes", flush=True)
    if jobs:
        with Pool(workers) as p, open(ck, 'a') as fh:
            for cid, core, unk, secs in p.imap_unordered(_core, jobs, chunksize=1):
                d = {'clase': cid, 'K': reps[cid]['ondas_min'], 'tam_clase': size[cid],
                     'nucleo': core, 'desconocidas': unk, 'seg': round(secs, 1)}
                fh.write(json.dumps(d) + '\n'); fh.flush(); os.fsync(fh.fileno())
                done[cid] = d
                print(f"  [{len([c for c in chosen if c in done])}/{len(chosen)}] clase {cid}: K={d['K']} nucleo {len(core)}"
                      f"{' (+%d sin resolver)' % len(unk) if unk else ''}  {secs:.0f}s", flush=True)
    tab = defaultdict(Counter)
    for c in chosen:
        d = done[c]
        tag = len(d['nucleo']) if not d['desconocidas'] else f">={len(d['nucleo'])}?"
        tab[d['K']][tag] += d['tam_clase']
    print(f"N={N}: tamano del nucleo -> soluciones (ponderado por tamano de clase)")
    for K in sorted(tab):
        rig = tab[K].get(K, 0); tot = sum(tab[K].values())
        print(f"   K={K}: {dict(sorted(tab[K].items(), key=str))}   rigidas {rig}/{tot} = {100*rig/tot:.0f}%"
              f"   con algun nucleo {sum(v for k, v in tab[K].items() if k not in (0, '>=0?'))}/{tot}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--kmax', type=int, default=5)
    ap.add_argument('--time-limit', type=float, default=120)
    ap.add_argument('--muestra-por-K', type=int, default=0)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    for N in a.Ns:
        run(N, a.kmax, a.time_limit, a.muestra_por_K, a.workers)
