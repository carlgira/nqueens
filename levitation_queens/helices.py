"""
Helices de paso cualquiera en el cilindro (fila = altura, columna = angulo, modulo p).

Una helice c(r) = s*r + c0 con s real recorre todas las alturas; en la fila r atraviesa la casilla
floor(s*r + c0) (modulo p). Toca una reina si esa casilla es la de la reina. Con s entero es una recta
(espiral) de siempre; con s no entero es una "recta digital" (palabra sturmiana).
Se generan TODAS las helices distintas sobre p filas (celdas de la particion del plano (s, c0) por las
rectas s*r + c0 = entero: basta s = fracciones de denominador < p y puntos medios, y c0 en cada hueco).
Para cada solucion: minimo de helices que tocan todas las reinas (MILP) y helice con mas reinas.
Azar: permutaciones al azar.

Uso: ../venv/bin/python helices.py p [n_azar]
"""
import sys, json
from fractions import Fraction
import numpy as np
from collections import Counter
from curvas import mascaras, cobertura


def helices(p):
    F = sorted({Fraction(a, b) for b in range(1, p) for a in range(0, b)})
    S = F + [(x + y) / 2 for x, y in zip(F, F[1:] + [Fraction(1)])]
    r = np.arange(p); out = set()
    for s in S:
        fr = sorted({(-s * x) % 1 for x in range(p)}) + [Fraction(1)]
        Cs = fr[:-1] + [(x + y) / 2 for x, y in zip(fr, fr[1:])]
        for c in Cs:
            base = tuple(int((s * x + c) // 1) for x in range(p))
            for k in range(p):                                     # parte entera del paso
                out.add(tuple((base[x] + k * x) % p for x in range(p)))
    return np.array(sorted(out))


if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    W = helices(p)
    print(f"p={p}: {len(W)} helices distintas (salvo giro del cilindro)")
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    from ordenadas import Orbitas
    Ob = Orbitas(p); expl = {Ob.clave(g["rep"]) for g in json.load(open(f"data/ordenadas_p{p}.json"))}
    grupos = {"explicada": [], "SIN EXPLICAR": [], "azar": []}
    def mide(T):
        M = mascaras(T, W, p)
        return cobertura(M, p), max(bin(m).count('1') for m in M)
    for i, o in enumerate(O):
        g = "explicada" if Ob.clave(o["rep"]) in expl else "SIN EXPLICAR"
        a, b = mide(o["rep"]); grupos[g].append((a, b))
        print(f"  {i}:c{o['clase']}x{o['tam']:<5d} {g:12s} minimo {a}  mejor helice {b} reinas", flush=True)
    rng = np.random.default_rng(0)
    for _ in range(nazar): grupos["azar"].append(mide(rng.permutation(p).tolist()))
    for g, L in grupos.items():
        print(f"  {g:12s} minimo: {dict(sorted(Counter(x[0] for x in L).items()))}   mejor helice: {dict(sorted(Counter(x[1] for x in L).items()))}")
