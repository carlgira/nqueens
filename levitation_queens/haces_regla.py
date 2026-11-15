"""
Regla de los haces n+m: "doble reclamo".

Un haz n+m son n rectas paralelas A_i (pendiente a) y m paralelas B_j (pendiente b). Cada A_i corta a
cada B_j en una fila t_ij: hay n*m cruces. Visto desde un cruce, las demas filas se reparten en clases
de H(a,b) (clase de r - t_ij). Se comprueba, sobre el censo sin regla de ./haces:

 1. DISJUNTAS: en cada cruce, A_i y B_j no comparten ninguna clase.
    (Si E(a,b) ya es un grupo, p.ej. pendientes (i,-i), es solo reescribir "no se atacan": probado.)
 2. DOBLE RECLAMO: reclamo(A_i) = filas que caen en una clase tocada por A_i vista desde TODOS sus
    cruces (interseccion de clases enteras); igual para B_j. ¿Cada recta ocupa exactamente su reclamo?
    Con m = 1 hay un solo cruce por paralela y esto es la regla vieja de "clases enteras".
 3. Cuando 2 falla: ¿es porque se solapan los reclamos de dos paralelas?
 4. (solo 2+2) T = dos soluciones de dos espirales pegadas: filas de A_i u B_j por un lado y las de la
    otra pareja por otro. ¿Alguno de los dos trozos es un "bloque" (union de clases enteras alrededor
    de algun centro)?
 Azar: las mismas rectas con el mismo numero de filas cada una, repartidas al azar.

Uso: ../venv/bin/python haces_regla.py p n m [fichero]
"""
import sys, random
from itertools import combinations
from collections import defaultdict, Counter
from clase_m import subgrupo, E


def rectas(T, p, a, b, n, m):
    """offsets (la, lb) de las n+m rectas que cubren T (None si no se identifican)."""
    ua = Counter((T[r] - a * r) % p for r in range(p)); ub = Counter((T[r] - b * r) % p for r in range(p))
    for la in combinations(sorted(ua, key=lambda c: -ua[c])[:n + 4], n):
        for lb in combinations(sorted(ub, key=lambda c: -ub[c])[:m + 4], m):
            if all((T[r] - a * r) % p in la or (T[r] - b * r) % p in lb for r in range(p)):
                return list(la), list(lb)
    return None


def analiza(p, a, b, n, m, la, lb, RA, RB, H, cos):
    """RA[i], RB[j] = conjuntos de filas de cada recta. Devuelve (disjuntas, exacta, solape, cr)."""
    inv = pow((a - b) % p, p - 2, p)
    cr = [[(lb[j] - la[i]) * inv % p for j in range(m)] for i in range(n)]
    tA = [[{cos[(r - cr[i][j]) % p] for r in RA[i] if r != cr[i][j]} for j in range(m)] for i in range(n)]
    tB = [[{cos[(r - cr[i][j]) % p] for r in RB[j] if r != cr[i][j]} for j in range(m)] for i in range(n)]
    disj = not any(tA[i][j] & tB[i][j] for i in range(n) for j in range(m))
    # una fila que es justo un cruce de la recta no tiene clase desde ese cruce: ahi vale de comodin
    cA = [{r for r in range(p) if all(r == cr[i][j] or cos[(r - cr[i][j]) % p] in tA[i][j] for j in range(m))} for i in range(n)]
    cB = [{r for r in range(p) if all(r == cr[i][j] or cos[(r - cr[i][j]) % p] in tB[i][j] for i in range(n))} for j in range(m)]
    XA = [set(cr[i]) for i in range(n)]; XB = [{cr[i][j] for i in range(n)} for j in range(m)]
    exacta = all(cA[i] - XA[i] == RA[i] - XA[i] for i in range(n)) and all(cB[j] - XB[j] == RB[j] - XB[j] for j in range(m))
    X = set().union(*XA)
    solape = set()
    for i, k in combinations(range(n), 2): solape |= (cA[i] & cA[k]) - X
    for j, k in combinations(range(m), 2): solape |= (cB[j] & cB[k]) - X
    return disj, exacta, len(solape), cr


def es_bloque(D, p, H):
    """D es union de clases enteras de H alrededor de algun centro."""
    D = set(D)
    if not D or len(D) % len(H): return False
    for c in range(p):
        if c in D: continue
        if all((c + (r - c) * h) % p in D for r in D for h in H): return True
    return False


if __name__ == "__main__":
    p, n, m = map(int, sys.argv[1:4])
    fich = sys.argv[4] if len(sys.argv) > 4 else f"data/haces_base_p{p}_{n}_{m}.txt"
    por = defaultdict(list)
    for ln in open(fich):
        v = list(map(int, ln.split())); por[(v[0], v[1])].append(v[2:])
    rng = random.Random(0)
    res = defaultdict(Counter)
    for (a, b), Ts in sorted(por.items()):
        Eab = E(a, b, p); H = sorted(subgrupo(Eab, p)); h = len(H)
        cos = {x: min(x * g % p for g in H) for x in range(1, p)}
        R = res[(h, len(Eab) == h)]
        for T in Ts:
            R["total"] += 1
            L = rectas(T, p, a, b, n, m)
            if L is None: R["rectas no identificadas"] += 1; continue
            la, lb = L
            RA = [{r for r in range(p) if (T[r] - a * r) % p == c} for c in la]
            RB = [{r for r in range(p) if (T[r] - b * r) % p == c} for c in lb]
            disj, exacta, sol, cr = analiza(p, a, b, n, m, la, lb, RA, RB, H, cos)
            R["disjuntas"] += disj; R["exacta"] += exacta
            if not exacta: R["no exacta y con solape de paralelas"] += sol > 0; R[f"  filas en solape = {sol}"] += 1
            # azar: mismas rectas y mismos tamaños, filas barajadas
            filas = list(range(p)); rng.shuffle(filas)
            tam = [len(x) for x in RA + RB]
            trozos, k = [], 0
            for t in tam: trozos.append(set(filas[k:k + t])); k += t
            d2, e2, _, _ = analiza(p, a, b, n, m, la, lb, trozos[:n], trozos[n:], H, cos)
            R["azar: disjuntas"] += d2; R["azar: exacta"] += e2
            if n == 2 and m == 2:
                alguno = False
                for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    D = (RA[1 - i] | RB[1 - j]) - (RA[i] | RB[j])          # trozo que NO es de la pareja (i, j)
                    if es_bloque(D, p, H): alguno = True
                R["algun trozo es bloque"] += alguno
                D = set(rng.sample(range(p), h * rng.randint(1, (p - 1) // h - 1)))
                R["azar: bloque"] += es_bloque(D, p, H)
    print(f"p={p} haz {n}+{m} ({fich})")
    for (h, grupo), R in sorted(res.items()):
        print(f" |H|={h} ({(p - 1) // h} clases){' [E=H: disjuntas esta probado]' if grupo else ''}: {R.pop('total')} soluciones base")
        for k, v in R.items(): print(f"     {k}: {v}")
