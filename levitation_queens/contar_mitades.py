"""
Contar las soluciones de N reinas juntando MITADES (exacto, 100 % por construccion).

Plantilla fija: filas 0..h-1 (arriba) y h..N-1 (abajo), h = N // 2. Cada solucion se
descompone de forma unica en su mitad de arriba y su mitad de abajo, asi que

    Q(N) = n de parejas (mitad arriba, mitad abajo) compatibles

donde "mitad" = cualquier colocacion sin ataques en esas filas (no hace falta conocer ninguna
solucion). Compatibles: columnas complementarias y diagonales disjuntas.

Los numeros del juego:
    A = n de mitades de arriba,  B = n de mitades de abajo
    C = parejas con columnas complementarias (las que el indice deja pasar)
    Q = parejas que ademas no chocan en diagonal  (= soluciones)
    p_col = C / (A*B),  p_diag = Q / C

Uso: python contar_mitades.py 4 14
"""
import sys, time
from collections import defaultdict

Q_KNOWN = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712,
           14: 365596, 15: 2279184, 16: 14772512}


def halves(N, rows):
    """todas las colocaciones sin ataques en esas filas -> {(cols, d1, d2): cuantas}"""
    out = defaultdict(int)

    def rec(idx, cols, d1, d2):
        if idx == len(rows):
            out[(cols, d1, d2)] += 1
            return
        r = rows[idx]
        for c in range(N):
            a, b, d = 1 << c, 1 << (r + c), 1 << (r - c + N)
            if cols & a or d1 & b or d2 & d:
                continue
            rec(idx + 1, cols | a, d1 | b, d2 | d)
    rec(0, 0, 0, 0)
    return out


def count(N):
    h = N // 2
    top = halves(N, list(range(h)))
    bot = halves(N, list(range(h, N)))
    A, B = sum(top.values()), sum(bot.values())
    full = (1 << N) - 1
    index = defaultdict(list)
    for (cols, d1, d2), n in bot.items():
        index[cols].append((d1, d2, n))
    C = Q = 0
    for (cols, d1, d2), n in top.items():
        for e1, e2, m in index.get(full ^ cols, ()):
            C += n * m
            if not (d1 & e1) and not (d2 & e2):
                Q += n * m
    return h, A, B, C, Q


if __name__ == '__main__':
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4, 13)
    print(f"{'N':>3} {'filas':>7} | {'A arriba':>10} {'B abajo':>10} | {'C columnas ok':>14} | {'Q soluciones':>12} {'conocido':>9} | "
          f"{'p_col':>9} {'p_diag':>7} | seg")
    for N in range(lo, hi + 1):
        t0 = time.time()
        h, A, B, C, Q = count(N)
        ok = 'OK' if Q == Q_KNOWN.get(N) else '??'
        print(f"{N:3d} {h:3d}+{N-h:<3d} | {A:10d} {B:10d} | {C:14d} | {Q:12d} {Q_KNOWN.get(N, 0):9d} {ok} | "
              f"{C/(A*B):9.2e} {Q/C:7.4f} | {time.time()-t0:.0f}", flush=True)
