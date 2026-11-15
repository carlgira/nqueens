"""
Sintesis: ir DIRECTO a las soluciones con las ondas que predice la teoria.

Un "disparo" = elegir K ondas, amplitudes y fases al azar, sumar el campo en la red
N x N y dejar caer una particula por fila en la casilla mas brillante de esa fila.
Si las N columnas y las 2 diagonales quedan sin repetir, es una solucion.

Conjuntos de ondas que se comparan:
  teoria     : ondas no degeneradas (u, v, u+v, u-v != 0 mod N) = clases de recta x armonicos
  todas      : cualquier k de la rejilla (incluye las degeneradas)
  ruido      : campo aleatorio independiente por casilla (control: nivel del azar)

Salida: tasa de acierto, soluciones distintas encontradas y, cuando se conoce Q(N),
que fraccion de TODAS las soluciones aparece.

Uso: python generador.py 8 10 11 12 13 14 16 20 --disparos 200000     (disparos a ciegas)
     python generador.py 8 10 16 20 --afinar --K 3 --cadenas 2000 --pasos 1500
"""
import argparse, math, time
import numpy as np
import min_waves as mw

Q = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712,
     14: 365596, 15: 2279184, 16: 14772512}


def wave_sets(N):
    allk = [(u, v) for u in range(N) for v in range(N) if (u, v) != (0, 0)]
    allk = [k for k in allk if min(k, ((-k[0]) % N, (-k[1]) % N)) == k]
    nd = [k for k in allk if 0 not in (k[0] % N, k[1] % N, (k[0] + k[1]) % N, (k[0] - k[1]) % N)]
    return {'teoria': np.array(nd), 'todas': np.array(allk)}


def valid_rows(C):
    """C: (B, N) columna por fila -> mascara de soluciones validas."""
    B, N = C.shape
    I = np.arange(N)
    def uniq(M):
        S = np.sort(M, axis=1)
        return (np.diff(S, axis=1) != 0).all(axis=1)
    return uniq(C) & uniq(C + I) & uniq(C - I)


def shots(N, K, ks, B, rng):
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    idx = np.array([rng.choice(len(ks), K, replace=False) for _ in range(B)])      # (B, K)
    k = ks[idx]                                                                       # (B, K, 2)
    A = rng.uniform(.3, 1, (B, K, 1, 1))
    ph = rng.uniform(0, 2 * np.pi, (B, K, 1, 1))
    th = 2 * np.pi * (k[:, :, 0, None, None] * I + k[:, :, 1, None, None] * J) / N + ph
    F = (A * np.cos(th)).sum(1)                                                        # (B, N, N)
    return F.argmax(2)


def run(N, total, Kmax, rng, batch=4000):
    sets = wave_sets(N)
    print(f"N={N}  (Q(N) = {Q.get(N, '?')})  ondas teoria {len(sets['teoria'])} / todas {len(sets['todas'])}")
    for K in range(1, Kmax + 1):
        line = f"   K={K}:"
        for name in ('teoria', 'todas', 'ruido'):
            if name == 'ruido' and K > 1:
                continue
            found, hits, t0, n = set(), 0, time.time(), 0
            while n < total:
                b = min(batch, total - n)
                if name == 'ruido':
                    C = rng.random((b, N, N)).argmax(2)
                else:
                    if K > len(sets[name]):
                        break
                    C = shots(N, K, sets[name], b, rng)
                ok = valid_rows(C)
                hits += ok.sum()
                found.update(map(tuple, C[ok]))
                n += b
            frac = f" ({100*len(found)/Q[N]:.1f}% de Q)" if N in Q else ''
            line += f"  {name}: {hits/total:.2e} aciertos, {len(found)} distintas{frac} |"
        print(line + f" {time.time()-t0:.0f}s", flush=True)




# ---------------------------------------------------------------- afinar las fuentes
def conflicts(C):
    """pares en conflicto (columna o diagonal) de cada fila de C (B, N)."""
    N = C.shape[1]
    I = np.arange(N)
    tot = 0
    for M in (C, C + I, C - I):
        E = M[:, :, None] == M[:, None, :]
        tot = tot + (E.sum((1, 2)) - N) // 2
    return tot


def field_cols(N, k, A, ph):
    """k (B,K,2), A y ph (B,K) -> columna mas brillante por fila (B,N)."""
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    th = 2 * np.pi * (k[:, :, 0, None, None] * I + k[:, :, 1, None, None] * J) / N + ph[:, :, None, None]
    return (A[:, :, None, None] * np.cos(th)).sum(1).argmax(2)


def tune(N, K, ks, chains, steps, rng, sigma=0.35):
    """Busqueda local: perturbar fases/amplitudes, aceptar si no empeora (y a veces si empeora)."""
    idx = np.array([rng.choice(len(ks), K, replace=False) for _ in range(chains)])
    k = ks[idx]
    A = rng.uniform(.3, 1, (chains, K)); ph = rng.uniform(0, 2 * np.pi, (chains, K))
    C = field_cols(N, k, A, ph); E = conflicts(C)
    found = set()
    for s in range(steps):
        A2 = np.clip(A + rng.normal(0, sigma / 3, A.shape), .05, 1)
        ph2 = ph + rng.normal(0, sigma, ph.shape)
        # a veces cambiar una onda por otra de la teoria
        sw = rng.random(chains) < .05
        if sw.any():
            j = rng.integers(0, K, sw.sum())
            k = k.copy(); k[np.where(sw)[0], j] = ks[rng.integers(0, len(ks), sw.sum())]
        C2 = field_cols(N, k, A2, ph2); E2 = conflicts(C2)
        acc = (E2 <= E) | (rng.random(chains) < np.exp(-(E2 - E) / .4))
        A[acc], ph[acc], C[acc], E[acc] = A2[acc], ph2[acc], C2[acc], E2[acc]
        ok = E == 0
        if ok.any():
            found.update(map(tuple, C[ok]))
            # reiniciar las cadenas que ya encontraron algo, para buscar otras
            r = np.where(ok)[0]
            k[r] = ks[np.array([rng.choice(len(ks), K, replace=False) for _ in r])]
            A[r] = rng.uniform(.3, 1, (len(r), K)); ph[r] = rng.uniform(0, 2 * np.pi, (len(r), K))
            C[r] = field_cols(N, k[r], A[r], ph[r]); E[r] = conflicts(C[r])
    return found


def run_tune(N, K, chains, steps, rng):
    t0 = time.time()
    ks = wave_sets(N)['teoria']
    f = tune(N, K, ks, chains, steps, rng)
    tor = sum(len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N for s in f)
    frac = f" = {100*len(f)/Q[N]:.1f}% de Q({N})" if N in Q else ''
    print(f"   afinando K={K}: {len(f)} soluciones distintas{frac} (toroidales {tor}) "
          f"| {chains} cadenas x {steps} pasos, {time.time()-t0:.0f}s", flush=True)
    return f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--disparos', type=int, default=100000)
    ap.add_argument('--kmax', type=int, default=6)
    ap.add_argument('--afinar', action='store_true')
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--cadenas', type=int, default=2000)
    ap.add_argument('--pasos', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for N in a.Ns:
        if a.afinar:
            print(f"N={N}")
            run_tune(N, a.K, a.cadenas, a.pasos, rng)
        else:
            run(N, a.disparos, a.kmax, rng)
