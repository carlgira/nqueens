"""
Fusionar dos tableros con una ONDA-PUERTA a lo largo de las filas.

    sigma(i) = L1(i)   si la fila i esta "abierta":  (u*i mod N) en [x, x+w)
    sigma(i) = L2(i)   si no

L1, L2 = toroidales de un tramo (a1*i + c1, a2*i + c2).  La puerta es una onda de frecuencia u
que solo depende de la fila; abre las filas cuya fase cae en un arco de anchura w.
    u = 1  -> un bloque de filas consecutivas (ciclico)
    u = 2, 3, ... -> patrones periodicos (filas alternas, de tres en tres, ...)

Uso: python puerta.py 11 13 17
"""
import sys, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def valid(sol):
    N = len(sol)
    return len(set(sol)) == N and len({i + c for i, c in enumerate(sol)}) == N and len({i - c for i, c in enumerate(sol)}) == N


def tor(s):
    N = len(s)
    return len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


def build(N):
    lines = [tuple((a * i + c) % N for i in range(N)) for a in range(2, N - 1) for c in range(N)]
    out = {}
    for u in range(1, N // 2 + 1):
        ph = [(u * i) % N for i in range(N)]
        gates = set()
        for x in range(N):
            for w in range(1, N):
                gates.add(frozenset(i for i in range(N) if (ph[i] - x) % N < w))
        for G in gates:
            for L1 in lines:
                for L2 in lines:
                    if L1 == L2:
                        continue
                    s = tuple(L1[i] if i in G else L2[i] for i in range(N))
                    if s not in out and valid(s):
                        out[s] = u
    return out


if __name__ == '__main__':
    for N in [int(x) for x in sys.argv[1:]] or [11, 13]:
        S = build(N)
        nt = {s: u for s, u in S.items() if not tor(s)}
        byu = Counter(nt.values())
        extra = ''
        f = os.path.join(HERE, 'data', f'waves_N{N}_nodeg.json')
        if os.path.exists(f):
            K = {tuple(r['solucion']): r['ondas_min'] for r in json.load(open(f))['soluciones']}
            extra = f" | ondas minimas de esas: {dict(sorted(Counter(K[s] for s in nt).items()))}"
        print(f"N={N}: {len(S)} soluciones ({len(nt)} no toroidales) | por frecuencia de la puerta (la menor que sirve): {dict(sorted(byu.items()))}{extra}", flush=True)
        json.dump([list(s) for s in nt], open(os.path.join(HERE, 'data', f'puerta_N{N}.json'), 'w'))
