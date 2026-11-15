"""
N primo: ¿ayuda la estructura (clase de recta, armonico) a generar soluciones?

Toda onda util es k = t*(1, s) mod N con s fuera de {0, inf, 1, -1}. Las pendientes se
agrupan en clases {s, -s, 1/s, -1/s}. Dos formas de elegir las K ondas a afinar:
  azar   : K ondas no degeneradas cualesquiera (como generador.py)
  receta : repartidas entre clases distintas (round-robin en orden aleatorio), cada
           una con pendiente y armonico al azar dentro de su clase, sin repetir recta
Se cuentan solo soluciones NO toroidales (las toroidales ya se construyen con 1 onda).

Uso: python primos.py 11 13 17 19 23 --K 3 4
"""
import argparse, math, time
import numpy as np
from generador import field_cols, conflicts, valid_rows

def classes(N):
    S = [s for s in range(2, N - 1)]
    seen, out = set(), []
    for s in S:
        if s in seen: continue
        c = sorted({s, (-s) % N, pow(s, -1, N), (-pow(s, -1, N)) % N})
        seen |= set(c); out.append(c)
    return out

def wave(s, t, N):
    return (t % N, (t * s) % N)

def pick(N, K, mode, rng, cls):
    if mode == 'azar':
        allw = [(s, t) for c in cls for s in c for t in range(1, (N + 1) // 2)]
        ch = rng.choice(len(allw), K, replace=False)
        return np.array([wave(*allw[i], N) for i in ch])
    order = rng.permutation(len(cls)); used = set(); ks = []
    j = 0
    while len(ks) < K:
        c = cls[order[j % len(cls)]]; j += 1
        s = int(rng.choice(c)); t = int(rng.integers(1, (N + 1) // 2))
        if (s, t) in used: continue
        used.add((s, t)); ks.append(wave(s, t, N))
    return np.array(ks)

def toroidal(sol, N):
    return len({(i + c) % N for i, c in enumerate(sol)}) == N and len({(i - c) % N for i, c in enumerate(sol)}) == N

def tune(N, K, mode, chains, steps, rng, sigma=.35):
    cls = classes(N)
    k = np.array([pick(N, K, mode, rng, cls) for _ in range(chains)])
    A = rng.uniform(.3, 1, (chains, K)); ph = rng.uniform(0, 2 * np.pi, (chains, K))
    C = field_cols(N, k, A, ph); E = conflicts(C); found = set()
    for _ in range(steps):
        A2 = np.clip(A + rng.normal(0, sigma / 3, A.shape), .05, 1); ph2 = ph + rng.normal(0, sigma, ph.shape)
        sw = rng.random(chains) < .05
        if sw.any():
            k = k.copy()
            for b in np.where(sw)[0]:
                k[b] = pick(N, K, mode, rng, cls)       # cambiar el juego de ondas entero, con la misma regla
        C2 = field_cols(N, k, A2, ph2); E2 = conflicts(C2)
        acc = (E2 <= E) | (rng.random(chains) < np.exp(-(E2 - E) / .4))
        A[acc], ph[acc], C[acc], E[acc] = A2[acc], ph2[acc], C2[acc], E2[acc]
        ok = np.where(E == 0)[0]
        if len(ok):
            found.update(tuple(int(x) for x in C[b]) for b in ok)
            for b in ok:
                k[b] = pick(N, K, mode, rng, cls)
            A[ok] = rng.uniform(.3, 1, (len(ok), K)); ph[ok] = rng.uniform(0, 2 * np.pi, (len(ok), K))
            C[ok] = field_cols(N, k[ok], A[ok], ph[ok]); E[ok] = conflicts(C[ok])
    return {s for s in found if not toroidal(s, N)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+'); ap.add_argument('--K', type=int, nargs='+', default=[3])
    ap.add_argument('--cadenas', type=int, default=2000); ap.add_argument('--pasos', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    for N in a.Ns:
        print(f"N={N}: {len(classes(N))} clases de recta {classes(N)}")
        for K in a.K:
            r = {}
            for mode in ('azar', 'receta'):
                t0 = time.time(); f = tune(N, K, mode, a.cadenas, a.pasos, rng)
                if f: assert valid_rows(np.array(list(f))).all()
                r[mode] = (len(f), time.time() - t0)
            print(f"   K={K}: no toroidales encontradas  azar {r['azar'][0]:4d}  |  receta {r['receta'][0]:4d}"
                  f"   ({r['azar'][1]:.0f}s / {r['receta'][1]:.0f}s)", flush=True)
