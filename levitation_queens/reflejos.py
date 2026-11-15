"""
Regla unificada (N = p primo): toroidal de un tramo + un conjunto simetrico de filas REFLEJADO.

    sigma(i) = a*i + c                 si i no esta en R
    sigma(i) = a*(m - i) + c           si i esta en R          (todo mod p)

R es simetrico respecto al centro m:  i en R  <=>  (m - i) mod p en R.
Reflejar una fila equivale a darle la pendiente -a (con otro desplazamiento).

Casos particulares ya conocidos:
    R = {0, p-1}, m = p-1 ............ extremos desplazados (familia H)
    R = bloque del borde ............. sube y baja (familia V)
    R = no residuos cuadraticos, m = 0  toroidal de pendientes +a/-a  (p = 1 mod 4)

Este script enumera TODOS los R simetricos (para p pequenos) y clasifica las soluciones.

Uso: python reflejos.py 7 11 13
"""
import itertools, sys
from collections import Counter, defaultdict


def valid(sol):
    N = len(sol)
    return len(set(sol)) == N and len({i + c for i, c in enumerate(sol)}) == N and len({i - c for i, c in enumerate(sol)}) == N


def tor(s):
    N = len(s)
    return len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


def reflect(p, a, c, m, R):
    return tuple((a * ((m - i) % p) + c) % p if i in R else (a * i + c) % p for i in range(p))


def all_reflections(p):
    """-> {solucion: (a, c, m, R)} con R de tamano minimo"""
    out = {}
    for m in range(p):
        pairs = sorted({tuple(sorted((i, (m - i) % p))) for i in range(p) if i != (m - i) % p})
        for mask in range(1, 1 << len(pairs)):
            R = frozenset(x for b, pr in enumerate(pairs) if mask >> b & 1 for x in pr)
            for a in range(2, p - 1):
                # el reflejo solo depende de c por traslacion de columnas: probar todas
                for c in range(p):
                    s = reflect(p, a, c, m, R)
                    if valid(s):
                        if s not in out or len(R) < len(out[s][3]):
                            out[s] = (a, c, m, R)
    return out


def shape_of(R, p):
    r = sorted(R)
    if len(r) == 2:
        return 'par de filas'
    runs = 1 + sum(r[i + 1] != r[i] + 1 for i in range(len(r) - 1))
    if runs == 1 and (r[0] == 0 or r[-1] == p - 1):
        return 'bloque del borde'
    if runs == 1:
        return 'bloque interior'
    if runs == 2 and (r[0] == 0 and r[-1] == p - 1):
        return 'dos bordes'
    return f'{runs} trozos'


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]] or [7, 11, 13]:
        out = all_reflections(p)
        T = sum(tor(s) for s in out)
        size = Counter(len(v[3]) for v in out.values())
        shp = defaultdict(Counter)
        for s, (a, c, m, R) in out.items():
            shp['toroidal' if tor(s) else 'no toroidal'][shape_of(R, p)] += 1
        print(f"p={p}: {len(out)} soluciones por reflejo ({T} toroidales, {len(out)-T} no toroidales)")
        print(f"   filas reflejadas -> soluciones: {dict(sorted(size.items()))}")
        for k, c in shp.items():
            print(f"   {k}: forma del conjunto reflejado {dict(c.most_common())}")


# ---------------------------------------------------------------- criterio exacto (caso toroidal)
def orbits(p, a, m):
    """componentes de Z_p bajo:  i -> m - i,  i -> k+(i),  i -> k-(i)   (companeras de diagonal)
       k+(i) = ((1-a) i + a m)/(1+a),   k-(i) = ((1+a) i - a m)/(1-a)        (mod p)"""
    inv = lambda x: pow(x % p, p - 2, p)
    kp = lambda i: ((1 - a) * i + a * m) * inv(1 + a) % p
    km = lambda i: ((1 + a) * i - a * m) * inv(1 - a) % p
    parent = list(range(p))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(p):
        for j in ((m - i) % p, kp(i), km(i)):
            parent[find(i)] = find(j)
    comp = defaultdict(list)
    for i in range(p):
        comp[find(i)].append(i)
    return list(comp.values())


def toroidal_reflections(p, store=True):
    """todas las toroidales 'un tramo + reflejo', construidas SIN validar: R = union de orbitas"""
    out = set()
    n = 0
    for a in range(2, p - 1):
        for m in range(p):
            comps = orbits(p, a, m)
            for mask in range(1 << len(comps)):
                R = frozenset(x for b, cc in enumerate(comps) if mask >> b & 1 for x in cc)
                n += 1
                if store:
                    for c in range(p):
                        out.add(reflect(p, a, c, m, R))
    return out


# ---------------------------------------------------------------- regla exacta general (toroidal o no)
def obligations(p, a, m, c):
    """Grafo de obligaciones entre 'unidades' (pares simetricos {i, m-i}).
    Devuelve (units, edges, forbidden):
       units[u]   = filas de la unidad u
       edges[u]   = unidades que hay que reflejar si se refleja u
       forbidden  = unidades que NO se pueden reflejar (chocan de verdad con la fila fija)
    Una fila reflejada i (columna u_i = a(m-i)+c) coincide en diagonal, modulo p, con UNA fila
    k del tramo original por cada diagonal. Si k no se refleja, hay ataque real  <=>  la
    coincidencia tambien se da como enteros (no 'da la vuelta')."""
    inv = lambda x: pow(x % p, p - 2, p)
    unit_of = {}
    units = []
    fixed = None
    for i in range(p):
        j = (m - i) % p
        if i == j:
            fixed = i
            continue
        if i not in unit_of:
            unit_of[i] = unit_of[j] = len(units)
            units.append((i, j))
    edges = [set() for _ in units]
    forbidden = set()
    for i in range(p):
        if i == fixed:
            continue
        ui = (a * ((m - i) % p) + c) % p
        for k in (((1 - a) * i + a * m) * inv(1 + a) % p, ((1 + a) * i - a * m) * inv(1 - a) % p):
            wk = (a * k + c) % p
            real = (i + ui == k + wk) or (i - ui == k - wk)       # ataque como enteros
            if not real:
                continue
            if k == fixed:
                forbidden.add(unit_of[i])
            elif unit_of[k] != unit_of[i]:
                edges[unit_of[i]].add(unit_of[k])
            else:
                forbidden.add(unit_of[i])
    return units, edges, forbidden


def closed_sets(n, edges, forbidden):
    """todos los conjuntos de unidades cerrados bajo las obligaciones y sin prohibidas.
    Enumeracion sin callejones sin salida (cada rama termina en un conjunto valido)."""
    preds = [set() for _ in range(n)]
    for u in range(n):
        for v in edges[u]:
            preds[v].add(u)

    def fwd(x):
        seen, st = {x}, [x]
        while st:
            y = st.pop()
            for z in edges[y]:
                if z not in seen:
                    seen.add(z); st.append(z)
        return seen

    def back(x):
        seen, st = {x}, [x]
        while st:
            y = st.pop()
            for z in preds[y]:
                if z not in seen:
                    seen.add(z); st.append(z)
        return seen
    out0 = set()
    for f in forbidden:
        out0 |= back(f)
    res = []

    def rec(u, inn, out):
        while u < n and (u in inn or u in out):
            u += 1
        if u == n:
            res.append(frozenset(inn)); return
        rec(u + 1, inn, out | back(u))             # no reflejar u (ni nada que obligue a u)
        rec(u + 1, inn | fwd(u), out)              # reflejar u y todo lo que obliga
    rec(0, set(), out0)
    return res


def all_by_rule(p):
    """todas las soluciones 'un tramo + reflejo', toroidales o no, SIN validar ninguna"""
    out = set()
    for a in range(2, p - 1):
        for m in range(p):
            for c in range(p):
                units, edges, forb = obligations(p, a, m, c)
                for S in closed_sets(len(units), edges, forb):
                    R = frozenset(x for u in S for x in units[u])
                    out.add(reflect(p, a, c, m, R))
    return out
