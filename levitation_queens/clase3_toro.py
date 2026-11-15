"""
Toroidales de clase 3: rectas con pendientes (a, a, -a), a^2 = -1 (mod p)  (solo p = 1 mod 4).
Por traslacion se fijan A1: a*x, B: -a*x; la segunda paralela A2: a*x + c, c = 1..p-1.
Se cuentan las asignaciones fila -> recta que son toroidales y usan las tres rectas; x p^2.
Uso: python clase3_toro.py 13 17 29 37 41
"""
import sys


def count(p):
    roots = [a for a in range(2, p - 1) if a * a % p == p - 1]
    total = 0; per_c = {}
    for a in roots:
        for c in range(1, p):
            L = [[(a * x) % p for x in range(p)], [(a * x + c) % p for x in range(p)], [(-a * x) % p for x in range(p)]]
            n = 0
            st = [(0, 0, 0, 0, 0)]
            while st:
                x, cols, s, d, used = st.pop()
                if x == p:
                    n += used == 7; continue
                seen = set()
                for k in range(3):
                    v = L[k][x]
                    if v in seen:
                        continue
                    seen.add(v)
                    kk = 0
                    for j in range(3):
                        if L[j][x] == v:
                            kk |= 1 << j
                    b, e, g = 1 << v, 1 << ((x + v) % p), 1 << ((x - v) % p)
                    if cols & b or s & e or d & g:
                        continue
                    # una fila sobre varias rectas cuenta para todas; "used" exige que cada recta tenga
                    # alguna fila que solo ella cubre (si no, bastarian dos rectas)
                    st.append((x + 1, cols | b, s | e, d | g, used | (kk if kk in (1, 2, 4) else 0)))
            per_c[(a, c)] = n; total += n
    return total, per_c


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]] or [13, 17]:
        if p % 4 != 1:
            print(f"p={p}: p = 3 mod 4 -> no hay a con a^2 = -1 -> 0"); continue
        t, pc = count(p)
        t //= 2                       # cada solucion sale dos veces (cualquiera de las dos paralelas puede ir por el origen)
        nz = {k: v for k, v in pc.items() if v}
        print(f"p={p}: por el origen {t} -> total {t * p * p}   (c con soluciones: {sorted(c for (a, c) in nz)[:12]}..., {len(nz)} pares (a,c))", flush=True)
