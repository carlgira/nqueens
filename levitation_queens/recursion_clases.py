"""
¿Se construye la clase m a partir de las clases menores?  (m = n minimo de rectas, minrectas.c)

Para cada solucion T de la clase m:
  (a) "una recta mas": existe A de clase < m que coincide con T fuera de las filas de UNA recta
      (T = A con un grupo de filas colineales cambiado)                     -> T = A + toroidal
  (b) "fusion libre": existen A y B de clases < m tales que T = A en unas filas y B en el resto
      (cualquier reparto de filas, sin plantilla de onda)
Se mira con las mascaras de coincidencia de T con todas las soluciones de clase < m.

Uso: python recursion_clases.py 11 13 [17]
"""
import sys
from collections import Counter, defaultdict
import numpy as np
MUESTRA = 3000


def load(p):
    cls, S = [], []
    for line in open(f'data/clases_p{p}.txt'):
        v = list(map(int, line.split()))
        cls.append(v[0]); S.append(v[1:])
    return np.array(cls), np.array(S, dtype=np.int16)


def collinear_sets(t, p):
    """mascaras de filas que estan en una misma recta (pendiente 2..p-2), con >= 2 filas"""
    out = set()
    r = np.arange(p)
    for a in range(2, p - 1):
        c = (t - a * r) % p
        for v in np.unique(c):
            m = int(sum(1 << i for i in np.where(c == v)[0]))
            out.add(m)
    return out


def run(p, maxm):
    cls, S = load(p)
    full = (1 << p) - 1
    W = (1 << np.arange(p)).astype(np.int64)
    print(f"===== p={p}: " + "  ".join(f"clase {m}: {(cls == m).sum()}" for m in range(1, cls.max() + 1)))
    for m in range(2, maxm + 1):
        low = S[cls < m]; T = S[cls == m]
        if not len(T):
            continue
        a_ok = b_ok = c_ok = 0
        if len(T) > MUESTRA:
            T = T[np.random.default_rng(0).choice(len(T), MUESTRA, replace=False)]
        for t in T:
            eq = ((low == t) * W).sum(1)                  # mascara de filas en que coincide con cada A
            ms = np.unique(eq)
            ms = ms[ms != 0]
            # (a): las filas en que difiere estan todas en una misma recta de T
            col = collinear_sets(t, p)
            a = any(((full ^ int(x)) & ~cm) == 0 for x in ms for cm in col)
            # (b): dos mascaras que juntas cubren todas las filas
            mm = sorted(ms.tolist(), key=lambda x: -bin(x).count('1'))
            b = False
            for i, x in enumerate(mm):
                need = full ^ x
                for y in mm[i:]:
                    if need & ~y == 0:
                        b = True; break
                if b or bin(x).count('1') * 2 < p:
                    break
            # (c): tres piezas menores (solo mascaras maximales)
            c = b
            if not b:
                mx = [x for x in mm if not any(y != x and (x & ~y) == 0 for y in mm)]
                c = any((full & ~(x | y | z)) == 0 for i, x in enumerate(mx) for j, y in enumerate(mx[i:], i) for z in mx[j:])
            a_ok += a; b_ok += b; c_ok += c
        n = len(T)
        print(f"   clase {m} ({n}): (a) A + una recta: {a_ok} ({100*a_ok/n:.1f}%) | (b) fusion libre de dos menores: {b_ok} ({100*b_ok/n:.1f}%)"
              f" | (c) hasta tres menores: {c_ok} ({100*c_ok/n:.1f}%)", flush=True)


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]] or [11, 13]:
        run(p, 5 if p <= 13 else 4)
    print(f'(clases con mas de {MUESTRA} soluciones: muestra al azar de {MUESTRA})')
