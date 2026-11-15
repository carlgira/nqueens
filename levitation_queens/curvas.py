"""
Otras "espirales": curvas que pasan una vez por cada fila pero solo tocan algunas reinas.

Para una familia de curvas c = c0 + w(r) (modulo p) se calcula el MINIMO numero de curvas de la
familia que tocan todas las reinas de T (cobertura exacta, MILP), y la curva que mas reinas toca.
Familias:
  rectas       w = a*r                               (las espirales de siempre: da la "clase")
  parabolas    w = a*r^2 + b*r
  potencias    w = a*(r - r0)^k, k = 2..p-2           (espirales de la rejilla de potencias; k = p-2 es
                                                       la hiperbola a/(r - r0))
  exponencial  w = a*g^r
  todas        union de las cuatro
Azar: permutaciones al azar (no son solucion). Una geometria "entiende" una solucion si la cubre con
claramente menos curvas que al azar.

Uso: ../venv/bin/python curvas.py p [n_azar]
"""
import sys, json
import numpy as np
from collections import Counter
from scipy.optimize import milp, LinearConstraint, Bounds


def formas(p):
    r = np.arange(p); F = {}
    F["rectas"] = [a * r % p for a in range(1, p)]
    F["parabolas"] = [(a * r * r + b * r) % p for a in range(1, p) for b in range(p)]
    F["potencias"] = [a * np.array([pow(int(x), k, p) for x in (r - r0) % p]) % p
                      for k in range(2, p - 1) for r0 in range(p) for a in range(1, p)]
    F["exponencial"] = [a * np.array([pow(g, int(x), p) for x in r]) % p for g in range(2, p) for a in range(1, p)]
    for k in F: F[k] = np.unique(np.array(F[k]), axis=0)
    F["todas"] = np.unique(np.concatenate([F[k] for k in ("rectas", "parabolas", "potencias", "exponencial")]), axis=0)
    return F


def mascaras(T, W, p):
    D = (np.asarray(T)[None, :] - W) % p                  # c0 de cada fila para cada forma
    pes = 1 << np.arange(p, dtype=np.int64); out = set()
    for c in range(p):
        m = ((D == c) * pes).sum(axis=1)
        out.update(int(x) for x in np.unique(m) if x & (x - 1))   # al menos 2 reinas
    out.update(1 << i for i in range(p))
    return list(out)


def cobertura(M, p):
    A = np.array([[(m >> i) & 1 for m in M] for i in range(p)])
    r = milp(np.ones(len(M)), constraints=LinearConstraint(A, lb=1), bounds=Bounds(0, 1), integrality=np.ones(len(M)))
    return int(round(r.fun))


if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    F = formas(p)
    print(f"p={p}: formas distintas por familia " + ", ".join(f"{k} {len(v)}" for k, v in F.items()))
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    from ordenadas import Orbitas
    Ob = Orbitas(p); expl = {Ob.clave(g["rep"]) for g in json.load(open(f"data/ordenadas_p{p}.json"))}
    fams = list(F)
    print(f"  {'solucion':24s} " + "  ".join(f"{k:>11s}" for k in fams) + "    (minimo de curvas / mas reinas en una curva)")
    def fila(T):
        res = []
        for k in fams:
            M = mascaras(T, F[k], p)
            res.append((cobertura(M, p), max(bin(m).count('1') for m in M)))
        return res
    grupos = {"explicada": [], "SIN EXPLICAR": [], "azar": []}
    for i, o in enumerate(O):
        g = "explicada" if Ob.clave(o["rep"]) in expl else "SIN EXPLICAR"
        res = fila(o["rep"]); grupos[g].append(res)
        print(f"  {i}:c{o['clase']}x{o['tam']:<5d} {g:12s} " + "  ".join(f"{a:>8d}/{b:<2d}" for a, b in res), flush=True)
    rng = np.random.default_rng(0)
    for j in range(nazar): grupos["azar"].append(fila(rng.permutation(p).tolist()))
    print("  resumen: reparto del minimo de curvas")
    for g, L in grupos.items():
        print(f"    {g:12s} " + "  ".join(f"{k}: {dict(sorted(Counter(x[n][0] for x in L).items()))}" for n, k in enumerate(fams)))
    print("  resumen: reparto de las reinas en la mejor curva")
    for g, L in grupos.items():
        print(f"    {g:12s} " + "  ".join(f"{k}: {dict(sorted(Counter(x[n][1] for x in L).items()))}" for n, k in enumerate(fams)))
