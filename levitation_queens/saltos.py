"""
¿De que N viene la base que se completa?  y  ¿cuanto salta de 10 a 11?

A) N' = 9, 10 (minimos exactos conocidos): para cada clase con K ondas y cada
   N1 < N' por separado, ¿algun conjunto de K-1 ondas de N1 (+1 onda nueva) la
   levita?  -> matriz N1 x K.

B) N' = 11: para cada clase, el conjunto heredado MAS PEQUENO (1..MAXB ondas, de
   cualquier N1 <= 10) que con una onda nueva la levita, y de que N1 viene.
   Total = base + 1  es una cota superior del minimo en N=11.

C) N' = 12, 13, 14 sobre una MUESTRA aleatoria de soluciones: para cada N1 por
   separado (y, si N1 divide a N', tambien con k escalado por N'/N1), ¿alguna base
   de 1..MAXB ondas + 1 onda nueva la levita?

Uso: python saltos.py A        python saltos.py B [--maxb 3]
     python saltos.py C --Ns 12 13 14 --n 100
Salida B: data/saltos_N11.json
"""
import argparse, json, os, time
from collections import Counter, defaultdict
from multiprocessing import Pool
from transfer import lp_margin, DELTA, LS
import min_waves as mw


def load(N):
    return json.load(open(os.path.join(mw.HERE, 'data', f'waves_N{N}.json')))['soluciones']


def norm(k, N):
    k = (k[0] % N, k[1] % N)
    return min(k, ((-k[0]) % N, (-k[1]) % N))


def bases_by(N2, Ns):
    """(N1, tamano) -> conjuntos orientados (k mod N2), sin duplicados."""
    out = defaultdict(set)
    for N1 in Ns:
        for r in load(N1):
            ks = [tuple(w['k']) for w in r['ondas']]
            for a, b, c, d in LS:
                o = tuple(sorted({norm((a * u + b * v, c * u + d * v), N2) for u, v in ks}))
                if len(o) == len(ks) and (0, 0) not in o:
                    out[(N1, len(ks))].add(o)
    return {k: sorted(v) for k, v in out.items()}


def first_completion(rep, N, bases, extras):
    for base in bases:
        for e in extras:
            if e in base:
                continue
            if lp_margin(rep, N, list(base) + [e])[0] >= DELTA:
                return list(base) + [e]
    return None


# ---------------------------------------------------------------- A
def _a(args):
    rep, N2, K, B, extras = args
    return {N1: first_completion(rep, N2, B.get((N1, K - 1), []), extras) is not None
            for N1 in range(4, N2)}


def part_a(workers):
    for N2 in (9, 10):
        t0 = time.time()
        B = bases_by(N2, range(4, N2))
        extras = mw.freq_pairs(N2)
        reps = {}
        for r in load(N2):
            if r['exacto'] and r['ondas_min'] >= 2:
                reps.setdefault(r['clase'], r)
        jobs = [(tuple(r['solucion']), N2, r['ondas_min'], B, extras) for r in reps.values()]
        with Pool(workers) as p:
            res = p.map(_a, jobs, chunksize=1)
        tab = Counter()
        nK = Counter(r['ondas_min'] for r in reps.values())
        for (_, _, K, _, _), d in zip(jobs, res):
            for N1, ok in d.items():
                tab[(K, N1)] += ok
        print(f"N'={N2} ({time.time()-t0:.0f}s): clases completables por N de origen (base de K-1 ondas)")
        print("   K  clases | " + " ".join(f"N1={n:<3d}" for n in range(4, N2)))
        for K in sorted(nK):
            print(f"   {K}  {nK[K]:5d}  | " + " ".join(
                f"{tab[(K, n)]:<6d}" if (n, K - 1) in B else f"{'-':<6s}" for n in range(4, N2)))


# ---------------------------------------------------------------- B
def _b(args):
    rep, N, B, extras, maxb = args
    for size in range(1, maxb + 1):
        found = {}
        for N1 in range(10, 3, -1):
            s = first_completion(rep, N, B.get((N1, size), []), extras)
            if s is not None:
                found[N1] = s
        if found:
            return size + 1, found
    return None, {}


def part_b(workers, maxb):
    t0 = time.time()
    N = 11
    B = bases_by(N, range(4, 11))
    extras = mw.freq_pairs(N)
    sols = mw.solutions(N)
    reps = mw.classes(sols, N)
    jobs = [(rep, N, B, extras, maxb) for rep in reps]
    res = []
    with Pool(workers) as p:
        for n, r in enumerate(p.imap(_b, jobs, chunksize=1), 1):
            res.append(r)
            if n % 50 == 0:
                print(f'  {n}/{len(reps)} clases, {time.time()-t0:.0f}s', flush=True)
    prev = {r['id']: r['ondas_cota'] for r in json.load(open(os.path.join(mw.HERE, 'data', 'transfer_N11.json')))['soluciones']}
    idx = {s: n for n, s in enumerate(sols)}
    recs, tot, orig, best = [], Counter(), Counter(), Counter()
    for cid, ((rep, members), (K, found)) in enumerate(zip(reps.items(), res)):
        m = len(members)
        p = prev[idx[rep]]
        tot[K or 0] += m
        for N1 in found:
            orig[N1] += m
        cands = [x for x in (K, p) if x]
        best[min(cands) if cands else 0] += m
        recs.append({'clase': cid, 'representante': list(rep), 'miembros': m,
                     'ondas_base_mas_1': K, 'origenes': {str(k): v for k, v in found.items()},
                     'ondas_transfer_directo': p})
    json.dump({'N': N, 'maxb': maxb, 'clases': recs},
              open(os.path.join(mw.HERE, 'data', 'saltos_N11.json'), 'w'), indent=1)
    print(f"N=11, bases de 1..{maxb} ondas + 1 nueva ({time.time()-t0:.0f}s)")
    print("  ondas (base+1) -> soluciones:", dict(sorted(tot.items())), " [0 = ninguna]")
    print("  soluciones completables por N de origen (una clase puede venir de varios):", dict(sorted(orig.items())))
    print("  mejor entre transfer directo y base+1 -> soluciones:", dict(sorted(best.items())))
    print(f"  cubiertas: {sum(v for k, v in best.items() if k)} de {len(sols)}")


# ---------------------------------------------------------------- C
def bases_scaled(N2, N1):
    """Solo si N1 divide a N2: k * (N2/N1), la version 'geometrica' del multiplo."""
    f = N2 // N1
    out = defaultdict(set)
    for r in load(N1):
        ks = [tuple(w['k']) for w in r['ondas']]
        for a, b, c, d in LS:
            o = tuple(sorted({norm((f * (a * u + b * v), f * (c * u + d * v)), N2) for u, v in ks}))
            if len(o) == len(ks) and (0, 0) not in o:
                out[len(ks)].add(o)
    return {k: sorted(v) for k, v in out.items()}


def _c(args):
    sol, N, fuentes, extras, maxb = args
    res = {}
    for name, B in fuentes.items():
        ok = None
        for size in range(1, maxb + 1):
            if first_completion(sol, N, B.get(size, []), extras) is not None:
                ok = size + 1
                break
        res[name] = ok
    return res


def random_solutions(N, n, rng):
    """Soluciones uniformes por rechazo sobre el arbol con backtracking aleatorio no es
    uniforme; aqui se enumera todo (N <= 14 cabe) y se muestrea."""
    sols = mw.solutions(N)
    return rng.sample(sols, n), len(sols)


def part_c(Ns, n, workers, maxb, seed=0):
    import random
    rng = random.Random(seed)
    for N in Ns:
        t0 = time.time()
        sample, total = random_solutions(N, n, rng)
        extras = mw.freq_pairs(N)
        fuentes = {}
        for N1 in range(4, 11):
            Bk = bases_by(N, [N1])
            fuentes[f'{N1}'] = {size: Bk[(N1, size)] for (m, size) in Bk}
            if N % N1 == 0:
                fuentes[f'{N1}x{N // N1}'] = bases_scaled(N, N1)
        with Pool(workers) as p:
            res = p.map(_c, [(s, N, fuentes, extras, maxb) for s in sample], chunksize=1)
        any_ok = sum(any(v for v in r.values()) for r in res)
        print(f"N={N}: muestra {n} de {total} soluciones, bases 1..{maxb} ondas + 1 nueva ({time.time()-t0:.0f}s)")
        print(f"   levitadas por alguna fuente: {any_ok}/{n} = {100*any_ok/n:.1f}%")
        for name in fuentes:
            k = sum(r[name] is not None for r in res)
            nb = sum(len(v) for v in fuentes[name].values())
            ondas = Counter(r[name] for r in res if r[name])
            print(f"   desde N={name:<5s} ({nb:4d} bases): {k:3d}  {dict(sorted(ondas.items()))}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('parte', choices=['A', 'B', 'C'])
    ap.add_argument('--Ns', type=int, nargs='+', default=[12, 13, 14])
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--maxb', type=int, default=3)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.parte == 'A':
        part_a(a.workers)
    elif a.parte == 'B':
        part_b(a.workers, a.maxb)
    else:
        part_c(a.Ns, a.n, a.workers, a.maxb)
