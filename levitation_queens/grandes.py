"""
N = 20 (4x5) y N = 25 (5x5): productos sigma(x)tau frente a soluciones "normales".

1. Productos: todos los sigma(x)tau con tau toroidal. Minimo de ondas (MILP exacto)
   dentro del conjunto candidato  {k_sigma + n*t} (alias) U {n*k_tau}.
2. Muestra de soluciones normales con min-conflicts aleatorio (no es uniforme).
3. ¿Levitan esas soluciones las ondas de los productos? ¿y las heredadas de N<=10?
   (LP, 8 orientaciones, solo amplitudes/fases/umbral).
4. MILP sobre TODAS las frecuencias con limite de tiempo, en unos pocos productos
   y unas pocas normales: cota superior (y inferior) del minimo real.

Uso: python grandes.py 20 25 [--muestra 200] [--milp 3 --tl 300]
Salida: data/grandes_N{N}.json
"""
import argparse, json, os, random, time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from transfer import lp_margin, params, DELTA, LS, load_supports
from multiples import load, product
import min_waves as mw

FACT = {20: (4, 5), 25: (5, 5)}


def norm(k, N):
    k = (k[0] % N, k[1] % N)
    return min(k, ((-k[0]) % N, (-k[1]) % N))


def signed(k, N):
    return [k[0] if k[0] <= N // 2 else k[0] - N, k[1] if k[1] <= N // 2 else k[1] - N]


# ---------------------------------------------------------------- min-conflicts
def min_conflicts(N, rng, maxit=200000):
    while True:
        x = list(range(N)); rng.shuffle(x)
        col = [1] * N; d1 = [0] * (2 * N); d2 = [0] * (2 * N)
        for i, j in enumerate(x):
            d1[i + j] += 1; d2[i - j + N] += 1
        for _ in range(maxit):
            bad = [i for i in range(N) if col[x[i]] + d1[i + x[i]] + d2[i - x[i] + N] > 3]
            if not bad:
                return tuple(x)
            i = rng.choice(bad)
            col[x[i]] -= 1; d1[i + x[i]] -= 1; d2[i - x[i] + N] -= 1
            c = [col[j] + d1[i + j] + d2[i - j + N] for j in range(N)]
            m = min(c)
            j = rng.choice([j for j in range(N) if c[j] == m])
            x[i] = j
            col[j] += 1; d1[i + j] += 1; d2[i - j + N] += 1
        # reinicio si no converge


def valid(p):
    N = len(p)
    return len(set(p)) == N and len({i + p[i] for i in range(N)}) == N and len({i - p[i] for i in range(N)}) == N


# ---------------------------------------------------------------- 1. productos
def _prod(args):
    rho, N, cand = args
    X, o, _ = mw.design(N, cand)
    sup, ex, lo = mw.min_support(rho, N, cand, X, o, 1e-3, 300)
    ks = [cand[i] for i in sup]
    waves, t, m = mw.best_params(rho, N, ks)
    return len(sup), ex, [signed(k, N) for k in ks], m


def productos(N, workers):
    n, m = FACT[N]
    S, T = load(n), [r for r in load(m) if r['ondas_min'] == 1]
    jobs = []
    for rs in S:
        for rt in T:
            rho = product(rs['solucion'], rt['solucion'])
            G = [tuple(w['k']) for w in rs['ondas']]
            F = [tuple(n * x for x in w['k']) for w in rt['ondas']]
            cand = []
            for k in [(g[0] + n * a, g[1] + n * b) for g in G for a in range(m) for b in range(m)] + F:
                k = norm(k, N)
                if k != (0, 0) and k not in cand:
                    cand.append(k)
            jobs.append((rho, N, cand))
    with Pool(workers) as p:
        res = p.map(_prod, jobs, chunksize=1)
    return [{'solucion': list(j[0]), 'ondas_min_en_alias': r[0], 'exacto': r[1], 'k': r[2], 'margen': r[3]}
            for j, r in zip(jobs, res)]


# ---------------------------------------------------------------- 3. transferencia a normales
SETS = None


def _init(s):
    global SETS
    SETS = s


def _hit(args):
    sol, N = args
    out = {}
    for name, sets in SETS.items():
        out[name] = None
        for ks in sets:
            for a, b, c, d in LS:
                kk = [(a * u + b * v, c * u + d * v) for u, v in ks]
                if lp_margin(sol, N, kk)[0] >= DELTA:
                    out[name] = len(ks)
                    break
            if out[name]:
                break
    return out


# ---------------------------------------------------------------- 4. MILP completo
def _full(args):
    sol, N, tl = args
    ks = mw.freq_pairs(N)
    X, o, _ = mw.design(N, ks)
    t0 = time.time()
    sup, ex, lo = mw.min_support(sol, N, ks, X, o, 1e-3, tl)
    return len(sup) if sup else None, ex, lo, round(time.time() - t0)


def run(N, n_muestra, n_milp, tl, workers):
    t0 = time.time()
    rng = random.Random(N)
    prods = productos(N, workers)
    h = Counter(p['ondas_min_en_alias'] for p in prods)
    print(f"N={N}: {len(prods)} productos sigma(x)tau | ondas (en alias) {dict(sorted(h.items()))} "
          f"| no exactos {sum(not p['exacto'] for p in prods)} | {time.time()-t0:.0f}s", flush=True)

    muestra = []
    seen = set()
    while len(muestra) < n_muestra:
        s = min_conflicts(N, rng)
        assert valid(s)
        if s not in seen:
            seen.add(s); muestra.append(s)
    es_prod = sum(tuple(s) in {tuple(p['solucion']) for p in prods} for s in muestra)
    print(f"   muestra: {len(muestra)} soluciones (min-conflicts), de ellas productos: {es_prod} "
          f"| {time.time()-t0:.0f}s", flush=True)

    sets_prod = sorted({tuple(sorted(tuple(k) for k in p['k'])) for p in prods})
    sets_inh = [ks for (esq, K, ks, orig) in load_supports(N) if esq == 'mismo_k']
    with Pool(workers, initializer=_init, initargs=({'productos': sets_prod, 'heredadas_N<=10': sets_inh},)) as p:
        hits = p.map(_hit, [(s, N) for s in muestra], chunksize=1)
    for name in ('productos', 'heredadas_N<=10'):
        c = Counter(hh[name] for hh in hits if hh[name])
        print(f"   ondas de {name:16s} ({len(sets_prod) if name == 'productos' else len(sets_inh)} conjuntos) "
              f"levitan {sum(c.values())}/{len(muestra)} normales {dict(sorted(c.items()))} | {time.time()-t0:.0f}s", flush=True)

    full = []
    if n_milp:
        pick = [('producto', tuple(p['solucion'])) for p in rng.sample(prods, n_milp)] + \
               [('normal', s) for s in muestra[:n_milp]]
        with Pool(min(workers, len(pick))) as p:
            res = p.map(_full, [(s, N, tl) for _, s in pick], chunksize=1)
        for (tipo, s), (K, ex, lo, secs) in zip(pick, res):
            full.append({'tipo': tipo, 'solucion': list(s), 'ondas': K, 'exacto': ex, 'cota_inferior': lo, 'seg': secs})
            print(f"   MILP completo {tipo:8s}: {K} ondas {'(exacto)' if ex else f'(cota; minimo >= {lo})'}  {secs}s", flush=True)

    json.dump({'N': N, 'productos': prods, 'muestra': [list(s) for s in muestra],
               'transfer_a_muestra': hits, 'milp_completo': full},
              open(os.path.join(mw.HERE, 'data', f'grandes_N{N}.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--muestra', type=int, default=200)
    ap.add_argument('--milp', type=int, default=3)
    ap.add_argument('--tl', type=float, default=300)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    for N in a.Ns:
        run(N, a.muestra, a.milp, a.tl, a.workers)
