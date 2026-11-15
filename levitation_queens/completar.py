"""
¿Una solucion de N' que necesita K ondas se obtiene COMPLETANDO con una onda mas
un conjunto de K-1 ondas heredado de un N menor?

Para cada clase de N' (minimo exacto K >= 2) se prueban todos los conjuntos de
K-1 ondas de N < N' (8 orientaciones, mismo k) mas cada onda extra posible de N'.
Control: el mismo numero de conjuntos de K-1 ondas elegidos AL AZAR entre las
frecuencias de N'. Si heredar no ayuda, las dos tasas deberian parecerse.

Uso: python completar.py 9 10
"""
import json, os, random, sys, time
from collections import Counter
from multiprocessing import Pool
from transfer import lp_margin, DELTA, LS
import min_waves as mw


def load(N):
    return json.load(open(os.path.join(mw.HERE, 'data', f'waves_N{N}.json')))['soluciones']


def norm(k, N):
    k = (k[0] % N, k[1] % N)
    return min(k, ((-k[0]) % N, (-k[1]) % N))


def completable(args):
    rep, N, bases, extras = args
    for base in bases:
        if len(set(base)) < len(base) or (0, 0) in base:
            continue
        for e in extras:
            if e in base:
                continue
            if lp_margin(rep, N, list(base) + [e])[0] >= DELTA:
                return True
    return False


def run(N2, workers=10, seed=0):
    t0 = time.time()
    recs = load(N2)
    extras = mw.freq_pairs(N2)
    reps = {}
    for r in recs:
        reps.setdefault(r['clase'], r)
    small = {}
    for N1 in range(4, N2):
        for r in load(N1):
            ks = [tuple(w['k']) for w in r['ondas']]
            for a, b, c, d in LS:
                o = tuple(sorted({norm((a * u + b * v, c * u + d * v), N2) for u, v in ks}))
                small.setdefault(len(ks), set()).add(o)
    rng = random.Random(seed)
    jobs_h, jobs_r, Ks = [], [], []
    for r in reps.values():
        K = r['ondas_min']
        if K < 2 or not r['exacto']:
            continue
        bases = sorted(small.get(K - 1, ()))
        rnd = [tuple(sorted(rng.sample(extras, K - 1))) for _ in bases]
        jobs_h.append((tuple(r['solucion']), N2, bases, extras))
        jobs_r.append((tuple(r['solucion']), N2, rnd, extras))
        Ks.append((K, len(bases)))
    with Pool(workers) as p:
        h = p.map(completable, jobs_h, chunksize=1)
        z = p.map(completable, jobs_r, chunksize=1)
    tab = Counter()
    for (K, nb), a, b in zip(Ks, h, z):
        tab[K] += 0
        tab[(K, 'n')] += 1
        tab[(K, 'her')] += a
        tab[(K, 'azar')] += b
        tab[(K, 'nb')] = nb
    print(f"N'={N2} ({time.time()-t0:.0f}s)")
    for K in sorted(k for k in tab if isinstance(k, int)):
        print(f"   K={K}: {tab[(K,'n')]} clases, {tab[(K,'nb')]} conjuntos de {K-1} ondas heredados | "
              f"completables heredando: {tab[(K,'her')]}   con conjuntos al azar: {tab[(K,'azar')]}", flush=True)


if __name__ == '__main__':
    for N in map(int, sys.argv[1:] or ['9', '10']):
        run(N)
