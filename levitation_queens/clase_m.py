"""
Coberturas minimas de clase >= 3 miradas "por parejas de rectas".
Para dos rectas de pendientes al, be que se cortan en la fila r0, una reina en la fila r de la primera
y otra en la fila r' de la segunda no se atacan  <=>  (r'-r0)/(r-r0) no esta en
    E(al,be) = {1, al/be, (al+1)/(be+1), (al-1)/(be-1)}.
H(al,be) = subgrupo generado por E (el mismo de la formula de la clase 2).
Se imprime, para cada pareja, |H| y si las filas de cada recta son clases enteras de H desde r0.
Uso: python clase_m.py p clase [-v]
"""
import sys, json
from collections import Counter
from orbitas import min_covers, inv

def subgrupo(gens, p):
    H = {1}; todo = [1]
    while todo:
        x = todo.pop()
        for g in gens:
            y = x * g % p
            if y not in H: H.add(y); todo.append(y)
    return H

def E(al, be, p):
    return {1, al * inv(be, p) % p, (al + 1) * inv(be + 1, p) % p, (al - 1) * inv(be - 1, p) % p}

def rows(m, p): return [r for r in range(p) if m >> r & 1]

def orden(x, p):
    n, y = 1, x
    while y != 1: y = y * x % p; n += 1
    return n

if __name__ == '__main__':
    p, cl = int(sys.argv[1]), int(sys.argv[2])
    for o in json.load(open(f"data/orbitas_p{p}.json")):
        if o['clase'] != cl: continue
        T = o['rep']; k, covs = min_covers(T, p)
        print(f"\np={p} clase {k}, orbita de tamano {o['tam']} (simetrias propias: {o['stab']}), {len(covs)} coberturas minimas   T={T}")
        for cv in covs[: (len(covs) if '-v' in sys.argv else 3)]:
            cv = sorted(cv)
            print("  rectas:", "   ".join(f"L{i}: a={a} c={c} filas={rows(m,p)}" for i, (a, c, m) in enumerate(cv)))
            for i in range(len(cv)):
                for j in range(i + 1, len(cv)):
                    (a1, c1, m1), (a2, c2, m2) = cv[i], cv[j]
                    if a1 == a2:
                        print(f"    L{i}-L{j}: paralelas, separacion de columnas {(c2-c1)%p}"); continue
                    r0 = (c2 - c1) * inv(a1 - a2, p) % p
                    H = subgrupo(E(a1, a2, p), p)
                    def enteras(m):
                        S = {(r - r0) % p for r in rows(m, p)} - {0}
                        return all({x * h % p for h in H} <= S for x in S)
                    # clases de H ocupadas por cada recta
                    def ocup(m):
                        S = {(r - r0) % p for r in rows(m, p)} - {0}
                        return len({min(x * h % p for h in H) for x in S})
                    print(f"    L{i}-L{j}: corte en fila {r0} (reina en el corte: {'si' if T[r0]==(a1*r0+c1)%p else 'no'}), |H|={len(H)} de {p-1};"
                          f" clases enteras: L{i} {'si' if enteras(m1) else 'no'}, L{j} {'si' if enteras(m2) else 'no'};"
                          f" clases tocadas {ocup(m1)}+{ocup(m2)} de {(p-1)//len(H)}")
