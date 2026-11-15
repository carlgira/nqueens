"""
Generador unico de las toroidales "ordenadas" de un primo p, sin recorrer tableros.

Familias (cada una construye soluciones directamente a partir de una regla):
  F0  una espiral          T = a*r                                   (formula)
  F1  dos espirales        clases de H(a,b) alrededor del cruce      (formula, si y solo si)
  F2  haces n+m            regla del doble reclamo: en cada cruce A_i y B_j no comparten clase de H;
                           solo parejas de pendientes con H != todo. La unica eleccion que queda es
                           que clases reclama cada recta en cada cruce.
  F3  centro de simetria   T(-r) = -T(r): una pendiente por cada pareja de filas {r, -r} (radial_impar.py)
  F4  mover bloques        desde todo lo anterior: liberar una union de clases alrededor de un centro
                           y recolocar esas reinas en sus mismas columnas y diagonales (bloques_mover.py),
                           repetido hasta que no sale nada nuevo.

Todo se cuenta salvo simetria (grupo completo 8(p-1) + traslaciones) y en orbitas de traslacion.
Validacion: cada solucion generada se comprueba como toroidal, y para p = 13, 17, 19 se compara con la
lista exhaustiva (data/orbitas_p{p}.json); para p = 23 con el volcado exhaustivo de clase <= 5
(data/clases_toro0_p23.txt) y el total 243.592.

Uso: ../venv/bin/python ordenadas.py p [max_soluciones_F4]
"""
import sys, json, os, time
from itertools import combinations
from collections import defaultdict, Counter
from orbitas import mats, apply
from haces import canon
from clase_m import subgrupo, E
from radial_impar import cuenta
from bloques_mover import bloques, recoloca


def es_toroidal(T, p):
    return len(set(T)) == p and len({(T[r] + r) % p for r in range(p)}) == p and len({(T[r] - r) % p for r in range(p)}) == p


def una_espiral(p):
    return [[a * r % p for r in range(p)] for a in range(2, p - 1)]


def dos_espirales(p):
    out = []
    for a in range(2, p - 1):
        for b in range(a + 1, p - 1):
            H = subgrupo(E(a, b, p), p)
            cls = sorted({tuple(sorted(x * h % p for h in H)) for x in range(1, p)})
            for k in range(1, len(cls)):
                for X in combinations(cls, k):
                    S = {x for c in X for x in c}
                    out.append([(b if r in S else a) * r % p for r in range(p)])
    return out


def haz(p, a, b, n, m):
    """haces n+m sobre (a, b) con la regla: en cada cruce las dos rectas no comparten clase de H."""
    H = sorted(subgrupo(E(a, b, p), p))
    cos = [0] + [min(x * g % p for g in H) for x in range(1, p)]
    inv = pow((a - b) % p, p - 2, p)
    out = []
    def lanza(la, lb):
        nl = n + m
        cr = [[(lb[j] - la[i]) * inv % p for j in range(m)] for i in range(n)]
        own = {}                                   # (i, j, clase) -> 'A' o 'B'
        T = [0] * p; usa = [0] * nl
        def rec(r, cols, sm, df):
            if r == p:
                if all(usa): out.append(list(T))
                return
            vistos = set()
            for L in range(nl):
                c = ((a * r + la[L]) if L < n else (b * r + lb[L - n])) % p
                if c in vistos: continue
                b1, b2, b3 = 1 << c, 1 << ((c + r) % p), 1 << ((c - r) % p)
                if cols & b1 or sm & b2 or df & b3: continue
                # la reina puede estar en un cruce (dos rectas a la vez): entonces reclama por las dos
                Ls = [M for M in range(nl) if (((a * r + la[M]) if M < n else (b * r + lb[M - n])) % p) == c]
                puestos, ok = [], True
                for M in Ls:
                    pares = [(M, j) for j in range(m)] if M < n else [(i, M - n) for i in range(n)]
                    for (i, j) in pares:
                        if r == cr[i][j]: continue
                        k = (i, j, cos[(r - cr[i][j]) % p]); q = 'A' if M < n else 'B'
                        if own.get(k, q) != q: ok = False; break
                        if k not in own: own[k] = q; puestos.append(k)
                    if not ok: break
                if ok:
                    vistos.add(c); T[r] = c
                    for M in Ls: usa[M] += 1
                    rec(r + 1, cols | b1, sm | b2, df | b3)
                    for M in Ls: usa[M] -= 1
                for k in puestos: del own[k]
        rec(0, 0, 0, 0)
    # colocaciones normalizadas (como haces.c): A_0 y B_0 por el origen, A_1 a distancia 1
    for restoA in combinations(range(2, p), n - 2) if n >= 2 else [()]:
        la = [0, 1][:n] + list(restoA)
        for restoB in combinations(range(1, p), m - 1):
            lanza(la, [0] + list(restoB))
    return out


def con_centro(p):
    _, sols, g = cuenta(p, guardar=True)
    out = []
    for ch in sols:
        T = [0] * p
        for j, a in enumerate(ch):
            x = pow(g, j, p); T[x] = a * x % p; T[p - x] = a * (p - x) % p
        out.append(T)
    return out


class Orbitas:
    """representantes salvo simetria completa; recuerda la familia que llego primero."""
    def __init__(self, p):
        self.p = p; self.Ms = mats(p); self.rep = {}; self.tam = {}; self.fam = {}; self.cache = {}
    def clave(self, T):
        t = canon(list(T), self.p)
        if t in self.cache: return self.cache[t]
        ims = {canon([int(x) for x in apply(M, t, self.p)], self.p) for M in self.Ms}
        k = min(ims)
        for u in ims: self.cache[u] = k
        self.tam[k] = len(ims)
        return k
    def mete(self, T, fam):
        assert es_toroidal(T, self.p), (fam, T)
        k = self.clave(T)
        if k not in self.fam: self.fam[k] = fam; return True
        return False


if __name__ == "__main__":
    p = int(sys.argv[1]); tope = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    t0 = time.time(); Ob = Orbitas(p); gen = Counter()
    def familia(nombre, sols):
        nuevas = sum(Ob.mete(T, nombre) for T in sols)
        gen[nombre] = len(sols)
        print(f"  {nombre:28s}: {len(sols):6d} construidas, {nuevas:4d} soluciones distintas nuevas  ({time.time()-t0:.0f} s)", flush=True)
    print(f"p={p}")
    familia("F0 una espiral", una_espiral(p))
    familia("F1 dos espirales", dos_espirales(p))
    pares = [(a, b) for a in range(2, p - 1) for b in range(2, p - 1) if a != b and len(subgrupo(E(a, b, p), p)) < p - 1]
    hs = Counter(len(subgrupo(E(a, b, p), p)) for a, b in pares if a < b)
    print(f"  parejas de pendientes con H pequeño: {dict(sorted(hs.items()))} (|H|: nº de parejas)")
    for n, m in ((2, 1), (3, 1), (2, 2), (4, 1), (3, 2)):
        sols = []
        for a, b in pares:
            if n == m and a > b: continue
            if (p - 1) // len(subgrupo(E(a, b, p), p)) < 3: continue      # con 2 clases no cabe un haz
            sols += haz(p, a, b, n, m)
        familia(f"F2 haz {n}+{m} (doble reclamo)", sols)
    familia("F3 centro de simetria", con_centro(p))
    # F4: cierre por movimientos de bloque
    B = list(bloques(p)); cola = list(Ob.fam); hechos = 0; nuevas = 0
    while cola and hechos < tope:
        T = cola.pop(0); hechos += 1
        for D in B:
            for U in recoloca(T, p, D):
                if Ob.mete(U, "F4 mover bloques"):
                    nuevas += 1; cola.append(Ob.clave(U))
    print(f"  {'F4 mover bloques':28s}: {hechos} soluciones exploradas, {nuevas:4d} distintas nuevas"
          f"{'' if not cola else f' (CORTADO: quedan {len(cola)} sin explorar)'}  ({time.time()-t0:.0f} s)")
    porfam = Counter(); orbfam = Counter()
    for k, f in Ob.fam.items(): porfam[f] += 1; orbfam[f] += Ob.tam[k]
    print("  resumen (soluciones distintas / orbitas de traslacion, sin repetir):")
    for f in porfam: print(f"     {f:28s}: {porfam[f]:5d} / {orbfam[f]}")
    tot = sum(orbfam.values())
    fo = f"data/orbitas_p{p}.json"
    if os.path.exists(fo):
        O = sorted(json.load(open(fo)), key=lambda o: (o["clase"], o["tam"]))
        claves = {Ob.clave(o["rep"]): i for i, o in enumerate(O)}
        assert all(k in claves for k in Ob.fam), "solucion generada que no esta en la lista exhaustiva"
        falta = [f"{i}:c{o['clase']}x{o['tam']}" for i, o in enumerate(O) if Ob.clave(o["rep"]) not in Ob.fam]
        T_ = sum(o["tam"] for o in O)
        print(f"  VALIDACION exhaustiva: generadas {len(Ob.fam)} de {len(O)} soluciones distintas = {tot} de {T_}"
              f" orbitas de traslacion ({100*tot/T_:.1f} %); todas las generadas estan en la lista")
        print(f"  no generadas: {falta}")
        for i, o in enumerate(O):
            k = Ob.clave(o["rep"])
            if k in Ob.fam: print(f"     {i}:c{o['clase']}x{o['tam']:<4d} <- {Ob.fam[k]}")
    else:
        print(f"  total generado: {len(Ob.fam)} soluciones distintas = {tot} orbitas de traslacion")
    json.dump([{"rep": list(k), "tam": Ob.tam[k], "familia": f} for k, f in Ob.fam.items()], open(f"data/ordenadas_p{p}.json", "w"))
