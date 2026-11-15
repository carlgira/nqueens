"""
Generador por bloques: partir de una toroidal, liberar un BLOQUE de filas y recolocar esas reinas.

Bloque = union de clases enteras de un subgrupo multiplicativo G (|G| divide a p-1, G != {1})
alrededor de una fila centro c:  D = c + X*G, con |D| <= (p-1)/2. El centro no se mueve.
Recolocar = las reinas de D se reparten de otra forma las mismas columnas y diagonales (no hay
busqueda fuera de D). Si sale otra toroidal se anota a que solucion distinta (salvo simetria) llega.

Preguntas:
 1. ¿Que soluciones se alcanzan encadenando movimientos de bloque desde las de una espiral?
 2. Azar: mismos tamaños, pero D = filas cualesquiera. ¿Cuantas veces sale algo?
 3. Para cada mezcla de dos (mezclas.py): ¿la pieza pequeña es un bloque?

Uso: ../venv/bin/python bloques_mover.py p [n_azar_por_solucion]
"""
import sys, json, random
from itertools import combinations
from collections import defaultdict, Counter
from intercambios import indice
from haces import canon
from clase_m import subgrupo


def subgrupos(p):
    g = next(g for g in range(2, p) if len({pow(g, e, p) for e in range(p - 1)}) == p - 1)
    return {d: sorted(subgrupo({pow(g, (p - 1) // d, p)}, p)) for d in range(2, p - 1) if (p - 1) % d == 0}


def bloques(p):
    """todos los bloques (frozenset de filas) -> lista de (centro, |G|, n clases)."""
    out = defaultdict(list)
    for d, G in subgrupos(p).items():
        cls = sorted({tuple(sorted(x * h % p for h in G)) for x in range(1, p)})
        for k in range(1, (p - 1) // (2 * d) + 1):
            if k * d < 4: continue
            for X in combinations(cls, k):
                base = [x for c in X for x in c]
                for c in range(p):
                    out[frozenset((c + x) % p for x in base)].append((c, d, k))
    return out


def recoloca(T, p, D):
    """todas las toroidales distintas de T que coinciden con T fuera de D y cambian TODAS las filas de D."""
    R = sorted(D); k = len(R)
    cols = [T[r] for r in R]
    fs = {(T[r] + r) % p for r in R}; fd = {(T[r] - r) % p for r in R}
    U = list(T); out = []
    def rec(i, usados, sm, df):
        if i == k: out.append(tuple(U)); return
        r = R[i]
        for j, c in enumerate(cols):
            if usados >> j & 1 or c == T[r]: continue
            s, d = (c + r) % p, (c - r) % p
            if s not in fs or d not in fd or s in sm or d in df: continue
            U[r] = c
            rec(i + 1, usados | 1 << j, sm | {s}, df | {d})
        U[r] = T[r]
    rec(0, 0, frozenset(), frozenset())
    return out


if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    idx = indice(p, O)
    nom = [f"{i}:c{o['clase']}x{o['tam']}" for i, o in enumerate(O)]
    B = bloques(p)
    tams = Counter(len(D) for D in B)
    print(f"p={p}: {len(B)} bloques distintos; por tamaño {dict(sorted(tams.items()))}")
    rng = random.Random(1)
    arcos = defaultdict(lambda: defaultdict(Counter))          # i -> j -> tipos de bloque
    porG = defaultdict(lambda: [0, 0])
    exito = Counter(); exito_azar = Counter(); prob = Counter(); prob_azar = Counter()
    for i, o in enumerate(O):
        T = o["rep"]
        for D, tipos in B.items():
            prob[len(D)] += 1
            sal = recoloca(T, p, D); g = max(t[1] for t in tipos)
            porG[g][0] += 1; porG[g][1] += bool(sal)
            if sal: exito[len(D)] += 1
            for U in sal:
                arcos[i][idx[canon(list(U), p)]][max(t[1] for t in tipos)] += 1
        for _ in range(nazar):
            k = rng.choice(list(tams.elements()))
            D = frozenset(rng.sample(range(p), k))
            if D in B: continue
            prob_azar[k] += 1
            if recoloca(T, p, D): exito_azar[k] += 1
        print(f"  {nom[i]:11s} -> " + (", ".join(f"{nom[j]} {dict(sorted(t.items()))}" for j, t in sorted(arcos[i].items())) or "nada"), flush=True)
    print("  bloques que dan otra toroidal, por tamaño (real | azar):")
    for k in sorted(tams):
        print(f"     {k} filas: {exito[k]}/{prob[k]} = {100*exito[k]/prob[k]:.2f} %  |  {exito_azar[k]}/{prob_azar[k]}"
              f" = {100*exito_azar[k]/max(1,prob_azar[k]):.2f} %")
    print("  por subgrupo mas grande del bloque (|G|: aciertos/probados):", {g: f"{b}/{a}" for g, (a, b) in sorted(porG.items())})
    # alcanzables desde las de una espiral
    visto = {i for i, o in enumerate(O) if o["clase"] == 1}; ronda = 0
    print(f"  ronda 0 (una espiral): {sorted(nom[i] for i in visto)}")
    while True:
        nuevo = {j for i in visto for j in arcos[i]} - visto
        if not nuevo: break
        ronda += 1; visto |= nuevo
        print(f"  ronda {ronda}: + {sorted((nom[j] for j in nuevo), key=lambda s: int(s.split(':')[0]))}")
    print(f"  alcanzadas {len(visto)} de {len(O)} soluciones distintas = {sum(O[i]['tam'] for i in visto)} de"
          f" {sum(o['tam'] for o in O)} orbitas de traslacion")
    print("  no alcanzadas:", [nom[i] for i in range(len(O)) if i not in visto])
    json.dump({"alcanzadas": sorted(visto), "arcos": {str(i): {str(j): {str(a): n for a, n in t.items()} for j, t in d.items()} for i, d in arcos.items()}},
              open(f"data/bloques_mover_p{p}.json", "w"))
