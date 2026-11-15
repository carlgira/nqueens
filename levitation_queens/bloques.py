"""
Soluciones por BLOQUES: g bloques iguales de paso s (N/g filas cada uno) unidos por g-1 saltos.

    g = 1: toroidal            g = 2: dos mitades (familia G)
    g = 3: tres tercios        g = 5: cinco quintos ...

Construccion por backtracking bloque a bloque (no se enumeran las Q(N) soluciones):
para cada columna inicial y cada salto se coloca el bloque entero y se poda al primer ataque.

Uso:
  python bloques.py --s 5 --g 5 --desde 10 --hasta 60      tabla N -> {saltos: n de inicios}
  python bloques.py --s 3 --g 3 --desde 6 --hasta 63
Salida: data/bloques_s{s}_g{g}.json
"""
import argparse, json, os, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def sg(x, N):
    x %= N
    return x if x <= N // 2 else x - N


def build(N, s, g):
    """-> Counter {(j1..j_{g-1}): n de columnas iniciales validas}"""
    h = N // g
    cols = [False] * N
    d1 = [False] * (2 * N)
    d2 = [False] * (2 * N)
    out = Counter()

    def place_block(b, c):
        """coloca el bloque b empezando en la columna c; devuelve la lista colocada o None"""
        done = []
        r0 = b * h
        for i in range(h):
            r = r0 + i
            x = (c + s * i) % N
            if cols[x] or d1[r + x] or d2[r - x + N]:
                for (rr, xx) in done:
                    cols[xx] = d1[rr + xx] = d2[rr - xx + N] = False
                return None
            cols[x] = d1[r + x] = d2[r - x + N] = True
            done.append((r, x))
        return done

    def undo(done):
        for (rr, xx) in done:
            cols[xx] = d1[rr + xx] = d2[rr - xx + N] = False

    def rec(b, last, jumps):
        if b == g:
            out[tuple(jumps)] += 1
            return
        for j in range(1, N):
            if j == s % N:
                continue                      # eso no seria un salto, seria seguir el tramo
            done = place_block(b, (last + j) % N)
            if done is None:
                continue
            jumps.append(j)
            rec(b + 1, done[-1][1], jumps)
            jumps.pop()
            undo(done)

    for c0 in range(N):
        done = place_block(0, c0)
        if done is None:
            continue
        rec(1, done[-1][1], [])
        undo(done)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--s', type=int, required=True)
    ap.add_argument('--g', type=int, required=True)
    ap.add_argument('--desde', type=int, default=0)
    ap.add_argument('--hasta', type=int, default=60)
    ap.add_argument('--max-seg', type=float, default=300)
    a = ap.parse_args()
    res = {}
    print(f"{a.g} bloques de paso {a.s}:  N -> total | saltos todos iguales {{j: n}} | n de combinaciones de saltos distintas")
    N = max(a.desde, 2 * a.g)
    N += (-N) % a.g
    while N <= a.hasta:
        t0 = time.time()
        r = build(N, a.s, a.g)
        res[N] = {','.join(map(str, k)): v for k, v in r.items()}
        eq = {k[0]: v for k, v in sorted(r.items()) if len(set(k)) == 1}
        dt = time.time() - t0
        print(f"N={N:3d} (mod 6 = {N%6}): {sum(r.values()):5d} | {eq} | {len(r)}   {dt:.1f}s", flush=True)
        json.dump(res, open(os.path.join(HERE, 'data', f'bloques_s{a.s}_g{a.g}.json'), 'w'))
        if dt > a.max_seg:
            break
        N += a.g
