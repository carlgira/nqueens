"""
Tabla de tipos: para cada orbita del grupo completo (clase >= 3) dice si alguna cobertura minima es
  HAZ     = solo dos pendientes (haces de rectas paralelas), p.ej. 2+1, 3+1, 2+2
  RADIAL  = todas las rectas pasan por una misma reina
y nombra las pendientes: i (i^2=-1), w (w^3=1, w!=1), s (= +-1+-raiz(2)), . (cualquier otra).
Uso: python tipos.py 17 19
"""
import sys, json
from collections import Counter, defaultdict
from orbitas import min_covers, inv
from clase_m import E, subgrupo

def nombre(a, p):
    if a * a % p == p - 1: return 'i'
    if a != 1 and pow(a, 3, p) == 1: return 'w'
    if a != p - 1 and pow(a, 3, p) == p - 1: return '-w'
    if (a * a - 2 * a - 1) % p == 0 or (a * a + 2 * a - 1) % p == 0: return 's'
    return '.'

def tipo(cv, T, p):
    sl = Counter(a for a, c, m in cv)
    if len(sl) == 2:
        (a1, n1), (a2, n2) = sorted(sl.items(), key=lambda x: -x[1])
        H = subgrupo(E(a1, a2, p), p)
        return f"HAZ {n1}+{n2} [{nombre(a1,p)},{nombre(a2,p)}] |H|={len(H)}"
    if len(sl) == len(cv):
        for r in range(p):
            if all(m >> r & 1 for a, c, m in cv):
                return "RADIAL [" + ','.join(sorted(nombre(a, p) for a in sl)) + "]"
    return None

if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]]:
        print(f"\np={p}   (orden del grupo sin traslaciones: {8*(p-1)})")
        res = defaultdict(list)
        for o in json.load(open(f"data/orbitas_p{p}.json")):
            if o['clase'] < 3: continue
            T = o['rep']; k, covs = min_covers(T, p)
            ts = Counter(t for t in (tipo(cv, T, p) for cv in covs) if t)
            impar = any(all(T[(2 * c - r) % p] == (2 * T[c] - T[r]) % p for r in range(p)) for c in range(p))
            res[k].append((o['tam'], len(covs), impar, ' / '.join(sorted(ts)) or 'otra'))
        for k in sorted(res):
            print(f"  clase {k}:")
            for tam, nc, impar, t in sorted(res[k], key=lambda x: (x[3], x[0])):
                print(f"     {tam:4d} orbitas de traslacion, {nc:3d} coberturas minimas, simetria central {'si' if impar else 'no'}:  {t}")
