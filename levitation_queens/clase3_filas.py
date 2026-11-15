"""
Clase 3 toroidal: que filas van en cada recta.
Rectas: B (pendiente -a), A1 y A2 (pendiente a), a^2 = -1.  r1, r2 = filas donde B corta a A1, A2.
Se normaliza con traslacion + escala para que r1 = 0, r2 = 1 y se imprime el reparto de filas
como "clases de mu4" (mu4 = las 4 raices cuartas de 1 mod p; la clase de x es el conjunto x*mu4).
"""
import sys, json
from collections import Counter, defaultdict
from orbitas import load, min_covers, inv

def clases(p):
    g = next(g for g in range(2, p) if len({pow(g, e, p) for e in range(p - 1)}) == p - 1)
    k = (p - 1) // 4
    lg = {pow(g, e, p): e % k for e in range(p - 1)}
    return g, k, lg

def normaliza(T, p):
    """devuelve (a, coloreo) con coloreo[r] en 'B','1','2','*'... tras llevar r1->0, r2->1."""
    k_, covs = min_covers(T, p)
    assert k_ == 3 and len(covs) == 1
    cv = covs[0]; sl = Counter(a for a, c, m in cv)
    a = next(s for s in sl if sl[s] == 2); B = next(l for l in cv if l[0] != a); A = [l for l in cv if l[0] == a]
    assert (a * a + 1) % p == 0 and (B[0] + a) % p == 0
    out = []
    for A1, A2 in (A, A[::-1]):
        r1 = (B[1] - A1[1]) * inv(2 * a, p) % p; r2 = (B[1] - A2[1]) * inv(2 * a, p) % p
        lam = inv(r2 - r1, p)
        col = [None] * p
        for r in range(p):
            s = ''.join(n for n, l in (('B', B), ('1', A1), ('2', A2)) if l[2] >> r & 1)
            col[(r - r1) * lam % p] = s
        out.append((a, tuple(col)))
    return out

if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]]:
        g, k, lg = clases(p)
        reps = [T for T, c in load(p).items() if c == 3]
        cols = Counter()
        for T in reps:
            for a, col in normaliza(T, p): cols[(a, col)] += 1
        print(f"p={p}: g={g}, {k} clases de mu4; {len(reps)} orbitas de traslacion; coloreos normalizados distintos: {len(cols)} (multiplicidades {sorted(set(cols.values()))})")
        for (a, col), n in sorted(cols.items()):
            def desc(name, centre):
                d = defaultdict(list)
                for r in range(p):
                    if name in col[r] and r != centre: d[lg[(r - centre) % p]].append(r)
                return ' '.join(f"C{j}:{len(v)}" for j, v in sorted(d.items()))
            print(f"  a={a}  {' '.join(f'{c:>2}' for c in col)}")
            print(f"      desde r1=0:  B[{desc('B',0)}]  A1[{desc('1',0)}]   | desde r2=1:  B[{desc('B',1)}]  A2[{desc('2',1)}]   fila0={col[0]} fila1={col[1]}")
