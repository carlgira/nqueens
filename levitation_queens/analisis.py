"""
Analisis de las ondas minimas, N = 4..11.

  1. numero de ondas vs N  (media, moda, maximo, ceil(N/2))
  2. direcciones: pendiente reducida a:b, orden de Farey max(a,b), angulo
  3. magnitud |k| / N  (separacion de franjas)
  4. relaciones entre las ondas de UNA solucion (armonicos, espejos, perpendiculares)
  5. parecido entre N (distancia entre distribuciones de direcciones)
  6. grupos: par / impar / primo

Fuentes: data/waves_N{4..10}.json (exactos) y data/waves_N11_nodeg.json (6 ondas = cota).
Uso: python analisis.py   (texto)  ->  y deja data/analisis.json para las figuras
"""
import json, math, os
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NS = list(range(4, 12))


def load(N):
    f = f'waves_N{N}.json' if N <= 10 else f'waves_N{N}_nodeg.json'
    return json.load(open(os.path.join(HERE, 'data', f)))['soluciones']


def isprime(n):
    return n > 1 and all(n % p for p in range(2, int(n ** .5) + 1))


def grupo(N):
    return 'primo' if isprime(N) else ('par' if N % 2 == 0 else 'impar compuesto')


def slope(u, v):
    """direccion salvo simetrias del cuadrado: (a, b) con 0 <= a <= b, gcd = 1."""
    a, b = sorted((abs(u), abs(v)))
    g = math.gcd(a, b) or 1
    return a // g, b // g, g


def fold_angle(u, v):
    """angulo en [0, 45] grados (las 8 simetrias del cuadrado lo pliegan ahi)."""
    a, b = sorted((abs(u), abs(v)))
    return math.degrees(math.atan2(a, b))


def main():
    D = {N: load(N) for N in NS}
    out = {}

    print("=" * 78)
    print("1. NUMERO DE ONDAS vs N")
    print(f"{'N':>3} {'grupo':>16} {'sol':>6} {'media':>6} {'moda':>5} {'max':>4} {'ceil(N/2)':>10} {'floor(N/2)':>10}  reparto")
    stats = {}
    for N, d in D.items():
        K = [r['ondas_min'] for r in d]
        c = Counter(K)
        moda = max(c, key=lambda k: (c[k], k))
        stats[N] = {'media': float(np.mean(K)), 'moda': moda, 'max': max(K), 'hist': dict(c)}
        print(f"{N:3d} {grupo(N):>16} {len(d):6d} {np.mean(K):6.2f} {moda:5d} {max(K):4d} "
              f"{math.ceil(N/2):10d} {N//2:10d}  {dict(sorted(c.items()))}")
    # sin toroidales (esas son 1 onda por construccion)
    print("   sin las toroidales:  N: media", {N: round(float(np.mean([r['ondas_min'] for r in d if r['ondas_min'] > 1])), 2) for N, d in D.items()})
    out['ondas'] = stats

    print("=" * 78)
    print("2. DIRECCIONES (pendiente reducida a:b, salvo simetria; % de todas las ondas de ese N)")
    dirs = {}
    allslopes = Counter()
    for N, d in D.items():
        c = Counter()
        for r in d:
            for w in r['ondas']:
                a, b, g = slope(*w['k'])
                c[(a, b)] += 1
        dirs[N] = c
        allslopes.update(c)
    top = [s for s, _ in allslopes.most_common(14)]
    print(f"{'N':>3} " + " ".join(f"{a}:{b:<3}" for a, b in top) + "  | Farey max(a,b): media, max")
    for N in NS:
        tot = sum(dirs[N].values())
        far = [max(a, b) for (a, b), v in dirs[N].items() for _ in range(v)]
        print(f"{N:3d} " + " ".join(f"{100*dirs[N][s]/tot:5.0f}" for s in top)
              + f"  | {np.mean(far):4.2f}, {max(far)}   (N/2 = {N/2})")
    out['direcciones'] = {N: {f'{a}:{b}': v for (a, b), v in c.items()} for N, c in dirs.items()}

    print("   ¿el orden de Farey maximo sigue a N?  max(a,b) usado vs floor(N/2):",
          {N: (max(max(a, b) for a, b in dirs[N]), N // 2) for N in NS})

    print("=" * 78)
    print("3. MAGNITUD |k|/N  (0.5 = franjas a 2 casillas; cuanto mayor, franjas mas juntas)")
    mags = {}
    for N, d in D.items():
        m = [math.hypot(*w['k']) / N for r in d for w in r['ondas']]
        mags[N] = m
        q = np.percentile(m, [10, 50, 90])
        print(f"{N:3d}  mediana {q[1]:.3f}  p10 {q[0]:.3f}  p90 {q[2]:.3f}   "
              f"fraccion con lambda < 2 casillas (no fisica): {np.mean(np.array(m) > .5):.0%}")
    out['magnitud'] = {N: m for N, m in mags.items()}

    print("=" * 78)
    print("4. RELACIONES ENTRE LAS ONDAS DE UNA MISMA SOLUCION (pares de ondas)")
    print(f"{'N':>3} {'pares':>6} {'misma dir':>9} {'espejo a:b<->b:a':>17} {'perpend.':>9} {'suma=otra':>10}")
    rel = {}
    for N, d in D.items():
        n = same = mirror = perp = summ = 0
        for r in d:
            ks = [tuple(w['k']) for w in r['ondas']]
            kset = {(u % N, v % N) for u, v in ks} | {((-u) % N, (-v) % N) for u, v in ks}
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    (u1, v1), (u2, v2) = ks[i], ks[j]
                    n += 1
                    s1, s2 = slope(u1, v1)[:2], slope(u2, v2)[:2]
                    if u1 * v2 - u2 * v1 == 0:
                        same += 1
                    elif s1 == s2:
                        mirror += 1          # misma pendiente salvo simetria, distinta direccion
                    if u1 * u2 + v1 * v2 == 0:
                        perp += 1
                    for sg in (1, -1):
                        k3 = ((u1 + sg * u2) % N, (v1 + sg * v2) % N)
                        if k3 in kset and k3 != (0, 0):
                            summ += 1
                            break
        rel[N] = dict(pares=n, misma=same, espejo=mirror, perp=perp, suma=summ)
        if n:
            print(f"{N:3d} {n:6d} {100*same/n:8.1f}% {100*mirror/n:16.1f}% {100*perp/n:8.1f}% {100*summ/n:9.1f}%")
    out['relaciones'] = rel

    print("=" * 78)
    print("5. PARECIDO ENTRE N  (1 - distancia de variacion total entre distribuciones de direcciones)")
    keys = sorted(allslopes)
    P = {N: np.array([dirs[N][s] for s in keys], float) / sum(dirs[N].values()) for N in NS}
    print("      " + " ".join(f"{N:5d}" for N in NS))
    sim = {}
    for A in NS:
        row = []
        for B in NS:
            s = 1 - 0.5 * np.abs(P[A] - P[B]).sum()
            sim[(A, B)] = s
            row.append(f"{s:5.2f}")
        print(f"{A:5d} " + " ".join(row))
    out['similitud'] = {f'{a},{b}': v for (a, b), v in sim.items()}

    print("=" * 78)
    print("6. GRUPOS  (par / impar compuesto / primo)")
    G = defaultdict(list)
    for N in NS:
        G[grupo(N)].append(N)
    for g, Ns in G.items():
        cop = Counter()
        tot = 0
        for N in Ns:
            for r in D[N]:
                for w in r['ondas']:
                    u, v = w['k']
                    cop[sum(math.gcd(p % N, N) == 1 for p in (u, v, u + v, u - v))] += 1
                    tot += 1
        within = [sim[(a, b)] for a in Ns for b in Ns if a < b]
        print(f"  {g:>16} {Ns}: proyecciones coprimas {dict(sorted((k, f'{100*v/tot:.0f}%') for k, v in cop.items()))}"
              + (f" | similitud media dentro del grupo {np.mean(within):.2f}" if within else ''))
    # similitud par-par vs impar-impar vs cruzada
    ev = [N for N in NS if N % 2 == 0]; od = [N for N in NS if N % 2]
    f = lambda X, Y: np.mean([sim[(a, b)] for a in X for b in Y if a != b])
    print(f"  similitud media: par-par {f(ev, ev):.2f} | impar-impar {f(od, od):.2f} | par-impar {f(ev, od):.2f}")
    print(f"  similitud N con N+1: {np.mean([sim[(N, N+1)] for N in NS[:-1]]):.2f} | con N+2: {np.mean([sim[(N, N+2)] for N in NS[:-2]]):.2f}")

    json.dump(out, open(os.path.join(HERE, 'data', 'analisis.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
