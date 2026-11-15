"""
Fusion A CIEGAS (sin lista de soluciones) con indice exacto de compatibilidad.

T = A en las filas abiertas (G) + B en las cerradas. T es solucion  <=>
    (1) columnas de B en las cerradas = las que le faltan a A en las abiertas   (igualdad)
    (2) las diagonales i+c de las dos partes son disjuntas
    (3) las diagonales i-c de las dos partes son disjuntas

Indice: para cada puerta, un diccionario  {mascara de columnas en las filas cerradas -> [B...]}.
Dada A se busca directamente la mascara complementaria; con las pocas B del cajon, (2) y (3)
son dos AND de enteros. Coste ~ |semilla| * puertas + pares del mismo cajon (no todos con todos).

Cada nivel anade tambien las 8 simetrias del tablero.

Uso: python fusion_ciega.py 13 --niveles 3
     python fusion_ciega.py 17 --niveles 2 --tope 200000
"""
import argparse, os, time, random
from collections import defaultdict
import min_waves as mw
from fusion import wave_gates

HERE = os.path.dirname(os.path.abspath(__file__))


def seed_rules(N):
    from reflejos import all_by_rule
    from desplazamientos import build, build3
    from escaleras import family, family2
    S = {tuple((a * i + c) % N for i in range(N)) for a in range(2, N - 1) for c in range(N)}
    S = {s for s in S if valid(s)}
    S |= set(family(N)) | set(family2(N)) | set(all_by_rule(N)) | set(build(N)) | set(build3(N, True))
    return S


def valid(s):
    N = len(s)
    return len(set(s)) == N and len({i + c for i, c in enumerate(s)}) == N and len({i - c for i, c in enumerate(s)}) == N


def close_sym(S, N):
    G8 = mw.board_maps(N)
    out = set()
    for s in S:
        for g in G8:
            out.add(mw.apply_map(g, s))
    return out


def fuse_halves(have, N, gates, tope=None):
    """Como fuse_level pero trabajando con MITADES distintas: para cada puerta se guardan las
    mitades abiertas y cerradas sin repetir; cada solucion resultante sale una sola vez por puerta."""
    full = (1 << N) - 1
    L = list(have)
    new = set()
    tested = hits = 0
    for G in gates:
        Gs = sorted(G)
        Cs = [i for i in range(N) if i not in G]
        closed = defaultdict(dict)            # mascara de columnas -> {mitad cerrada: (d1, d2)}
        opened = {}                           # mitad abierta -> (cols, e1, e2)
        for s in L:
            hc = tuple(s[i] for i in Cs)
            cm = d1 = d2 = 0
            for i, c in zip(Cs, hc):
                cm |= 1 << c; d1 |= 1 << (i + c); d2 |= 1 << (i - c + N)
            closed[cm][hc] = (d1, d2)
            hg = tuple(s[i] for i in Gs)
            if hg not in opened:
                cg = e1 = e2 = 0
                for i, c in zip(Gs, hg):
                    cg |= 1 << c; e1 |= 1 << (i + c); e2 |= 1 << (i - c + N)
                opened[hg] = (cg, e1, e2)
        for hg, (cg, e1, e2) in opened.items():
            bucket = closed.get(full ^ cg)
            if not bucket:
                continue
            for hc, (d1, d2) in bucket.items():
                tested += 1
                if (e1 & d1) or (e2 & d2):
                    continue
                hits += 1
                t = [0] * N
                for i, c in zip(Gs, hg):
                    t[i] = c
                for i, c in zip(Cs, hc):
                    t[i] = c
                t = tuple(t)
                if t not in have:
                    new.add(t)
        if tope and len(new) >= tope:
            break
    return new, tested, hits


def fuse_level(have, N, gates, tope=None, rng=None):
    """-> (nuevas, estadisticas). 'have' = conjunto de soluciones conocidas."""
    full = (1 << N) - 1
    L = list(have)
    new = set()
    tested = hits = 0
    for G in gates:
        Gs = sorted(G)
        Cs = [i for i in range(N) if i not in G]
        index = defaultdict(list)
        partG = []
        for s in L:
            cm = d1 = d2 = 0
            for i in Cs:
                c = s[i]
                cm |= 1 << c; d1 |= 1 << (i + c); d2 |= 1 << (i - c + N)
            index[cm].append((d1, d2, s))
            cg = e1 = e2 = 0
            for i in Gs:
                c = s[i]
                cg |= 1 << c; e1 |= 1 << (i + c); e2 |= 1 << (i - c + N)
            partG.append((cg, e1, e2, s))
        for cg, e1, e2, a in partG:
            bucket = index.get(full ^ cg)
            if not bucket:
                continue
            for d1, d2, b in bucket:
                tested += 1
                if (e1 & d1) or (e2 & d2):
                    continue
                hits += 1
                if b is a:
                    continue
                t = tuple(a[i] if i in G else b[i] for i in range(N))
                if t not in have:
                    new.add(t)
        if tope and len(new) >= tope:
            break
    return new, tested, hits


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--niveles', type=int, default=3)
    ap.add_argument('--tope', type=int, default=0, help='parar un nivel al superar este n de nuevas')
    ap.add_argument('--muestra', type=int, default=0, help='usar solo una muestra de lo conocido como piezas')
    ap.add_argument('--mitades', action='store_true', help='fusionar mitades distintas (sin redundancia por puerta)')
    a = ap.parse_args()
    N = a.N
    t0 = time.time()
    have = close_sym(seed_rules(N), N)
    gates = list(wave_gates(N))
    print(f"N={N}: nivel 0 (reglas + simetrias) = {len(have)} soluciones | {len(gates)} puertas | {time.time()-t0:.0f}s", flush=True)
    rng = random.Random(0)
    for lev in range(1, a.niveles + 1):
        t1 = time.time()
        pieces = have
        if a.muestra and len(have) > a.muestra:
            pieces = set(rng.sample(sorted(have), a.muestra))
        if a.mitades:
            new, tested, hits = fuse_halves(pieces, N, gates, a.tope or None)
        else:
            new, tested, hits = fuse_level(pieces, N, gates, a.tope or None, rng)
        new = close_sym(new, N) - have
        assert all(valid(s) for s in list(new)[:2000])
        naive = len(pieces) ** 2 * len(gates)
        print(f"   nivel {lev}: +{len(new)} -> {len(have)+len(new)} | pares comprobados {tested:,} (todos con todos serian {naive:,}: {naive/max(tested,1):,.0f}x mas) "
              f"| validos {hits:,} ({100*hits/max(tested,1):.0f}% de los comprobados) | {time.time()-t1:.0f}s", flush=True)
        if not new:
            break
        have |= new
