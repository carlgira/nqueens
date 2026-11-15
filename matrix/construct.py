"""Construcción explícita de una solución de N-reinas en O(N) (pares y luego impares, con ajuste según N mod 6).
Imprime la solución como lista de columnas (1..N) por fila y la guarda en qN.txt, lista para ./nq_cayley N query < qN.txt
Uso: python construct.py 8 9 10 ... 16
"""
import sys


def construct(n):
    ev = list(range(2, n + 1, 2)); od = list(range(1, n + 1, 2))
    r = n % 6
    if r == 2:                       # intercambiar 1 y 3 en los impares y mover 5 al final
        i1, i3 = od.index(1), od.index(3); od[i1], od[i3] = od[i3], od[i1]
        od.remove(5); od.append(5)
    elif r == 3:                     # mover 2 al final de los pares y 1,3 al final de los impares
        ev.remove(2); ev.append(2)
        od.remove(1); od.remove(3); od += [1, 3]
    return ev + od


def valid(q):
    n = len(q)
    return (sorted(q) == list(range(1, n + 1)) and len({q[i] + i for i in range(n)}) == n
            and len({q[i] - i for i in range(n)}) == n)


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        q = construct(n)
        assert valid(q), f"la construcción falla para N={n}"
        open(f"q{n}.txt", "w").write(",".join(map(str, q)) + "\n")
        print(n, ",".join(map(str, q)))
