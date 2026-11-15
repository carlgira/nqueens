"""
Clase 2 contada EXACTAMENTE con las espirales de super_queens.py.

Espiral (d, sentido, giro i)  <->  recta  T(r) = s*2d*(r+1) - i  (mod p),  s = +1 / -1.
(numeracion de super_queens.init_sols_label: indice = 26*(d-1) + 2*i + (0 si + / 1 si -), para p=13)

Dos espirales A, B. Para que mezclarlas de solucion, las filas que toman B (conjunto R) deben usar
las mismas columnas que A en esas filas  ->  R es union de CICLOS de la aplicacion
        pi(r) = A^-1(B(r)) = m*r + k  (mod p),   m = pendiente_B / pendiente_A
  - mismo sentido y misma velocidad (m = 1): un solo ciclo -> nunca mezclan
  - misma velocidad, sentido contrario (m = -1): parejas de filas simetricas respecto a la fila
    donde se cruzan -> (p-1)/2 parejas
  - en general: ciclos de longitud ord(m), (p-1)/ord(m) de ellos (+ la fila de cruce)
Solo queda comprobar las diagonales (no toroidales) de cada union de ciclos: es un recuento exacto,
sin busqueda por el tablero.

Uso: python espirales.py 11 13 17 19
"""
import sys
from itertools import combinations
from collections import Counter, defaultdict


def order(m, p):
    k, x = 1, m % p
    while x != 1:
        x = x * m % p; k += 1
    return k


def spiral(d, s, i, p):
    return [(s * 2 * d * (r + 1) - i) % p for r in range(p)]


def valid(t):
    p = len(t)
    return len(set(t)) == p and len({r + c for r, c in enumerate(t)}) == p and len({r - c for r, c in enumerate(t)}) == p


def spirals(p):
    out = []
    for d in range(1, p // 2):                     # como super_queens: 2d != +-1
        for i in range(p):
            for s in (1, -1):
                out.append(((d, s, i), spiral(d, s, i, p)))
    return out


def mix(A, B, p):
    """todas las soluciones que mezclan A y B (usando las dos), por union de ciclos"""
    inv = {v: r for r, v in enumerate(A)}
    pi = [inv[B[r]] for r in range(p)]
    seen, cycles = set(), []
    for r in range(p):
        if r in seen or pi[r] == r:
            seen.add(r); continue
        cyc = []; x = r
        while x not in seen:
            seen.add(x); cyc.append(x); x = pi[x]
        cycles.append(cyc)
    sols = []
    for k in range(1, len(cycles)):
        for chosen in combinations(range(len(cycles)), k):
            R = {r for c in chosen for r in cycles[c]}
            t = [B[r] if r in R else A[r] for r in range(p)]
            if valid(t):
                sols.append(tuple(t))
    return cycles, sols


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]] or [13]:
        S = spirals(p)
        total, cand = set(), 0
        bym = defaultdict(lambda: [0, 0, 0, 0])            # m -> [pares, pares que funcionan, candidatos, soluciones]
        for x in range(len(S)):
            for y in range(x + 1, len(S)):
                (la, A), (lb, B) = S[x], S[y]
                ma = (A[1] - A[0]) % p; mb = (B[1] - B[0]) % p
                m = mb * pow(ma, -1, p) % p
                cycles, sols = mix(A, B, p)
                key = 'misma (m=1)' if m == 1 else 'sentido contrario, misma velocidad (m=-1)' if m == p - 1 else f'orden {order(m, p)}'
                bym[key][0] += 1; bym[key][1] += bool(sols); bym[key][2] += 2 ** len(cycles) - 2 if cycles else 0; bym[key][3] += len(sols)
                total |= set(sols)
        print(f"===== p={p}: {len(S)} espirales, {len(total)} soluciones de clase 2")
        print(f"   {'relacion (m = pendiente B / pendiente A)':<44} | pares | pares utiles | candidatos | soluciones")
        for k, (n, u, c, s) in sorted(bym.items(), key=lambda kv: -kv[1][3]):
            print(f"   {k:<44} | {n:5d} | {u:12d} | {c:10d} | {s:10d}")
        if p == 13:
            lab = {i: l for i, (l, _) in enumerate(S)}
            A, B = S[15][1], S[22][1]
            cycles, sols = mix(A, B, p)
            print(f"\n   ejemplo: espiral 15 = {lab[15]} y espiral 22 = {lab[22]}  (d, sentido, giro)")
            print(f"   ciclos de filas: {cycles}")
            print(f"   uniones de ciclos que dan solucion: {len(sols)}")
            for t in sols:
                print("     ", t)
