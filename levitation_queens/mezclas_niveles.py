"""
¿Las toroidales sin descripcion salen mezclando las que SI tenemos?

Piezas permitidas por niveles acumulados (todas sus copias: traslacion, escala, giro):
  p=19: 1 rectas · 2 dos espirales · 3 con centro de simetria · 4 las obtenidas por mezcla
  p=17: 1 rectas · 2 dos espirales · 3 tres espirales · 4 cuatro espirales · 5 las dos con receta
Para cada objetivo: ¿es mezcla de dos / de tres piezas de nivel <= c?  Y lo mismo para permutaciones
al azar (modelo nulo), que es lo que dice si el resultado significa algo.

Uso: ../venv/bin/python mezclas_niveles.py p [n_azar]
"""
import sys, json
import numpy as np
from orbitas import mats, apply
from mezclas3 import analiza

NIVELES = {19: {1: [0, 1], 2: [2, 3], 3: [5, 6, 7, 8, 9, 10, 12], 4: [4, 11, 19, 20]},
           17: {1: [0, 1, 2], 2: [3, 4, 5, 6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15, 16, 17, 18], 5: [19, 21]}}
OBJ = {19: [13, 14, 15, 16, 17, 18, 21, 22], 17: [20, 22]}

if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    Ms = mats(p); S = []; cl = []
    for niv, ids in NIVELES[p].items():
        for i in ids:
            vistos = set()
            for M in Ms:
                U = apply(M, O[i]["rep"], p)
                for s in range(p):
                    V = np.roll(U, -s); V = (V - V[0]) % p
                    vistos.add(tuple(int(x) for x in V))
            for V in vistos:
                V = np.array(V)
                S.append((V[None, :] + np.arange(p)[:, None]) % p); cl += [niv] * p
    S = np.concatenate(S).astype(np.int16); cl = np.array(cl, dtype=np.int8); cmax = max(NIVELES[p])
    print(f"p={p}: piezas por nivel acumulado:", [int((cl <= c).sum()) for c in range(1, cmax + 1)])
    for i in OBJ[p]:
        r = analiza(O[i]["rep"], p, cl, S, cmax)
        c2 = next((c for c in r if r[c][1]), None); c3 = next((c for c in r if r[c][2]), None)
        tam = [bin(x).count("1") for x in r[c3][3]] if c3 and r[c3][3] else ""
        print(f"  solucion {i} ({O[i]['clase']} helices): dos piezas -> nivel {c2};  tres piezas -> nivel {c3} {tam}", flush=True)
    rng = np.random.default_rng(1); n2 = [0] * (cmax + 1); n3 = [0] * (cmax + 1)
    for _ in range(nazar):
        r = analiza(rng.permutation(p).tolist(), p, cl, S, cmax)
        for c in r: n2[c] += r[c][1]; n3[c] += r[c][2]
    print(f"  azar ({nazar} permutaciones) por nivel 1..{cmax}: dos piezas {n2[1:]}, tres piezas {n3[1:]}")
