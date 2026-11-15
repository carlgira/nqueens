"""
¿Una solucion sin simetrias es "casi" simetrica (una simetrica con unas pocas reinas rotas)?

Para cada solucion distinta de data/orbitas_p{p}.json se prueban todas las simetrias del toro
(8(p-1) giros/escalas x p^2 traslaciones, salvo la identidad) y se anota el maximo numero de reinas
que caen otra vez sobre reinas. p de p = simetria exacta. Azar: lo mismo con permutaciones al azar.

Uso: ../venv/bin/python casi_simetria.py p [n_azar]
"""
import sys, json, random
from collections import Counter
from orbitas import mats


def casi(T, p, Ms):
    Q = [(r, T[r]) for r in range(p)]; top = 0
    for M in Ms:
        (a, b), (c, d) = M
        if M == ((1, 0), (0, 1)): continue
        cnt = Counter()
        for r in range(p):
            R, C = (a * r + b * T[r]) % p, (c * r + d * T[r]) % p
            for (r2, c2) in Q: cnt[((r2 - R) % p, (c2 - C) % p)] += 1
        top = max(top, max(cnt.values()))
    return top


if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    Ms = mats(p)
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    for i, o in enumerate(O):
        n = casi(o["rep"], p, Ms)
        if n < p: print(f"  {i}:c{o['clase']}x{o['tam']}: mejor casi-simetria {n} de {p}")
    rng = random.Random(0); c = Counter()
    for _ in range(nazar):
        T = list(range(p)); rng.shuffle(T); c[casi(T, p, Ms)] += 1
    print(f"  p={p} azar ({nazar} permutaciones): {sorted(c.items())}")
