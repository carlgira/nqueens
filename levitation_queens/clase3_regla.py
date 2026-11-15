"""
Regla de la clase 3 toroidal (p = 1 mod 4), SIN busqueda fila a fila.
mu4 = {1, a, -1, -a} (a^2 = -1). Clases C_j = g^j * mu4, j = 0..k-1, k = (p-1)/4.
Con r1 = 0 y r2 = 1 (filas donde la recta B corta a las paralelas A1 y A2):
    A1 toma las filas  X*mu4        (X = conjunto de clases, medido desde r1)
    A2 toma las filas  1 + Y*mu4    (Y = conjunto de clases, medido desde r2)
    B toma todas las demas.
Condicion: ningun x en X*mu4 e y en Y*mu4 con x - y en mu4  ("X e Y no son vecinos").
Numero de soluciones = p^2 * (p-1) * N(p),  N(p) = numero de pares (X, Y) no vacios no vecinos.
Uso: python clase3_regla.py 13 17 29 37 41 [--check]   (--check: construye y valida cada solucion)
"""
import sys, itertools
from orbitas import inv

def datos(p):
    g = next(g for g in range(2, p) if len({pow(g, e, p) for e in range(p - 1)}) == p - 1)
    k = (p - 1) // 4
    mu4 = {pow(g, k * e, p) for e in range(4)}
    C = [{pow(g, j, p) * m % p for m in mu4} for j in range(k)]
    cls = {x: j for j in range(k) for x in C[j]}
    # vecinos[i] = mascara de las clases j con (C_i - C_j) que toca mu4
    vec = [0] * k
    for i in range(k):
        for x in C[i]:
            for m in mu4:
                y = (x - m) % p
                if y: vec[i] |= 1 << cls[y]
    return g, k, mu4, C, vec

def pares(p):
    g, k, mu4, C, vec = datos(p)
    N = 0; lista = []
    for X in range(1, 1 << k):
        nb = 0
        for i in range(k):
            if X >> i & 1: nb |= vec[i]
        libres = [j for j in range(k) if not nb >> j & 1]
        N += (1 << len(libres)) - 1
        lista.append((X, libres))
    return N, lista, (g, k, mu4, C, vec)

def construye(p, a, X, Y, C):
    """solucion con r1=0, r2=1: B: c = -a r ; A1: c = a r ; A2: c = a (r-1) - a = a r - 2a."""
    T = []
    A1 = set().union(*[C[i] for i in X]); A2 = {(1 + y) % p for j in Y for y in C[j]}
    for r in range(p):
        if r in A1: T.append(a * r % p)
        elif r in A2: T.append((a * r - 2 * a) % p)
        else: T.append(-a * r % p)
    return T

def toroidal(T, p):
    return all(len({(T[r] + e * r) % p for r in range(p)}) == p for e in (0, 1, -1))

if __name__ == '__main__':
    check = '--check' in sys.argv
    for p in [int(x) for x in sys.argv[1:] if x.isdigit()]:
        if p % 4 != 1: print(f"p={p}: 0 (p = 3 mod 4)"); continue
        N, lista, (g, k, mu4, C, vec) = pares(p)
        print(f"p={p}: k={k} clases, grados de vecindad {sorted(bin(v).count('1') for v in vec)};  N={N}  ->  por orbita de traslacion {(p-1)*N},  total {(p-1)*N*p*p}", flush=True)
        if check:
            from orbitas import min_covers, canon_tr
            a = next(x for x in mu4 if x not in (1, p - 1)); vistos = set(); ok = 0; cl = {}
            for X, libres in lista:
                Xs = [i for i in range(k) if X >> i & 1]
                for n in range(1, len(libres) + 1):
                    for Y in itertools.combinations(libres, n):
                        for aa in (a, p - a):
                            T = construye(p, aa, Xs, Y, C)
                            assert toroidal(T, p)
                            for lam in range(1, p):
                                U = [0] * p
                                for r in range(p): U[lam * r % p] = lam * T[r] % p
                                vistos.add(canon_tr(U, p))
                            if p <= 17:
                                m = min_covers(T, p)[0]; cl[m] = cl.get(m, 0) + 1
            print(f"   construidas: todas toroidales; orbitas de traslacion distintas = {len(vistos)}  (clases minimas: {cl})")
