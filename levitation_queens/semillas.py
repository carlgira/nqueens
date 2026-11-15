"""
Simetrias propias de cada toroidal distinta y cuantas reinas "semilla" hacen falta.

Para cada solucion de data/orbitas_p{p}.json: todas las simetrias del toro (giro M + traslacion)
que la dejan igual, y las orbitas en que esas simetrias agrupan a sus p reinas. Si hay s orbitas,
la solucion entera se reconstruye a partir de s reinas (las semillas) aplicando sus simetrias.

Uso: ../venv/bin/python semillas.py 17 19
"""
import sys, json
from collections import Counter
from orbitas import mats, inv


def simetrias(T, p, Ms):
    """lista de (M, s, t): M*(r, T[r]) + (s, t) recorre otra vez las reinas de T."""
    Q = {(r, T[r]) for r in range(p)}
    out = []
    for M in Ms:
        (a, b), (c, d) = M
        R0, C0 = (a * 0 + b * T[0]) % p, (c * 0 + d * T[0]) % p
        for r in range(p):                       # la imagen de la reina 0 cae sobre la reina r
            s, t = (r - R0) % p, (T[r] - C0) % p
            if all((((a * x + b * T[x]) + s) % p, ((c * x + d * T[x]) + t) % p) in Q for x in range(p)):
                out.append((M, s, t))
    return out


def semillas(T, p, sims):
    padre = list(range(p))
    def f(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]; x = padre[x]
        return x
    for ((a, b), (c, d)), s, t in sims:
        for x in range(p):
            padre[f(x)] = f((a * x + b * T[x] + s) % p)
    return sorted(Counter(f(x) for x in range(p)).values())


def tipo(M, p):
    (a, b), (c, d) = M
    if b == 0 and c == 0 and a == d:
        return "identidad" if a == 1 else "centro" if a == p - 1 else f"escala x{a}"
    return "giro"


if __name__ == "__main__":
    for p in map(int, sys.argv[1:]):
        Ms = mats(p)
        print(f"p={p}")
        tot = Counter()
        for o in sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"])):
            T = o["rep"]
            S = simetrias(T, p, Ms)
            orb = semillas(T, p, S)
            tipos = Counter(tipo(M, p).split(" ")[0] for M, s, t in S)
            print(f"  clase {o['clase']} x{o['tam']:3d}: {len(S):3d} simetrias ({dict(tipos)}) -> "
                  f"{len(orb)} semillas, grupos de reinas {orb}")
            tot[len(orb)] += o["tam"]
        print("  orbitas de traslacion por numero de semillas:", dict(sorted(tot.items())))
