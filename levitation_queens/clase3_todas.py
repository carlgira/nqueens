"""
Clase 3 SIN suponer las pendientes: prueba todas las ternas de rectas (salvo traslacion y escala)
y busca fila a fila las toroidales cubiertas por ellas. Sirve para comprobar en primos mayores que
la unica clase 3 es (a, a, -a) con a^2 = -1.
Uso: python clase3_todas.py 13 17 19 23 29
"""
import sys
from collections import Counter
from orbitas import canon_tr, inv

def clase_le2(T, p):
    for a in range(2, p - 1):
        cnt = Counter((T[r] - a * r) % p for r in range(p))
        c, n = cnt.most_common(1)[0]
        if n == p: return True
        if 2 * n >= p:
            rest = [r for r in range(p) if (T[r] - a * r) % p != c]
            if len(rest) <= 1: return True
            b = (T[rest[1]] - T[rest[0]]) * inv(rest[1] - rest[0], p) % p
            if all((T[r] - T[rest[0]] - b * (r - rest[0])) % p == 0 for r in rest): return True
    return False

def busca(p, lines):
    """toroidales con cada fila sobre alguna de las rectas (a, c)."""
    opts = [sorted({(a * r + c) % p for a, c in lines}) for r in range(p)]
    out = []; T = [0] * p
    def rec(r, cols, s, d):
        if r == p: out.append(tuple(T)); return
        for v in opts[r]:
            b, e, g = 1 << v, 1 << ((r + v) % p), 1 << ((r - v) % p)
            if cols & b or s & e or d & g: continue
            T[r] = v; rec(r + 1, cols | b, s | e, d | g)
    rec(0, 0, 0, 0)
    return out

def todas(p):
    S = range(2, p - 1); vistos = {}
    configs = []
    for al in S:
        for be in S:
            if be <= al: continue
            for ga in S:
                for c in ((1,) if ga in (al, be) else (0, 1)):
                    configs.append(((al, 0), (be, 0), (ga, c)))
        for t in range(2, p): configs.append(((al, 0), (al, 1), (al, t)))
    for lines in configs:
        for T in busca(p, lines):
            if clase_le2(T, p): continue
            key = min(canon_tr([lam * T[inv(lam, p) * r % p] % p for r in range(p)], p) for lam in range(1, p))
            if key not in vistos:
                orb = {canon_tr([lam * T[inv(lam, p) * r % p] % p for r in range(p)], p) for lam in range(1, p)}
                vistos[key] = (len(orb), tuple(sorted(a for a, c in lines)))
    return vistos

if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]]:
        v = todas(p)
        pats = Counter()
        for n, sl in v.values(): pats[sl] += n
        print(f"p={p}: clase 3 exacta = {sum(n for n, _ in v.values())} orbitas de traslacion; por pendientes: {dict(pats)}", flush=True)
