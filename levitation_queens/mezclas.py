"""
¿Cada toroidal es mezcla de otras dos?  T = A en unas filas y B en el resto.

Para cada solucion distinta T (data/orbitas_p{p}.json) se compara con TODAS las toroidales del
volcado (data/clases_toro_p{p}.txt): cada A coincide con T en un conjunto de filas. T es mezcla de
A y B si entre las dos cubren todas las filas. Se busca la mezcla con piezas de clase mas baja y,
dentro de esa, la mas equilibrada. Modelo nulo: lo mismo con permutaciones al azar como objetivo.

Uso: ../venv/bin/python mezclas.py p [n_azar]
"""
import sys, json
import numpy as np
from orbitas import mats, apply
from haces import canon


def carga(p):
    A = np.loadtxt(f"data/clases_toro_p{p}.txt", dtype=np.int16)
    return A[:, 0].astype(np.int8), A[:, 1:]


def superconjuntos(has, p):
    """sup[m] = hay alguna mascara presente que contiene a m."""
    sup = has.copy()
    idx = np.arange(1 << p)
    for b in range(p):
        sin = idx[(idx >> b) & 1 == 0]
        sup[sin] |= sup[sin | (1 << b)]
    return sup


def mezcla(T, p, cl, S, cmax):
    """(clase maxima de las piezas, mascara de A, mascara de B) o None."""
    full = (1 << p) - 1
    pesos = (1 << np.arange(p, dtype=np.int64))
    m = ((S == np.asarray(T)) * pesos).sum(axis=1)
    distinta = m != full
    for c in range(1, cmax + 1):
        sel = distinta & (cl <= c)
        has = np.zeros(1 << p, dtype=bool); has[m[sel]] = True
        sup = superconjuntos(has, p)
        ms = np.unique(m[sel])
        ok = ms[sup[full ^ ms]]
        if len(ok):
            # la mas equilibrada: maximizar el tamaño de la pieza pequeña
            pc = np.array([bin(int(x)).count("1") for x in ok])
            mejor = None
            for a in ok[np.argsort(-pc)][:200]:
                resto = full ^ int(a)
                cand = ms[(ms & resto) == resto]
                b = int(cand[np.argmin([bin(int(x)).count("1") for x in cand])])
                solo_b = bin(resto).count("1")               # filas que solo aporta B
                if mejor is None or solo_b < mejor[0]: mejor = (solo_b, int(a), b)
            return c, mejor[1], mejor[2], len(ok), m, sel
    return None


if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    cl, S = carga(p)
    cmax = int(cl.max())
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    Ms = mats(p); idx = {}
    for i, o in enumerate(O):
        for M in Ms: idx[canon([int(x) for x in apply(M, o["rep"], p)], p)] = i
    nom = [f"{i}:c{o['clase']}x{o['tam']}" for i, o in enumerate(O)]
    print(f"p={p}: {len(S)} toroidales")
    for i, o in enumerate(O):
        T = o["rep"]
        r = mezcla(T, p, cl, S, cmax)
        if r is None:
            print(f"  {nom[i]:12s}: NO es mezcla de dos toroidales"); continue
        c, a, b, n, m, sel = r
        ia = np.flatnonzero(sel & (m == a))[0]; ib = np.flatnonzero(sel & (m == b))[0]
        na = bin(a).count("1"); nb_solo = p - na
        print(f"  {nom[i]:12s}: mezcla de piezas de clase <= {c}:  {na} filas de {nom[idx[canon(S[ia].tolist(), p)]]}"
              f" + {nb_solo} de {nom[idx[canon(S[ib].tolist(), p)]]}   ({n} formas de elegir la pieza grande)", flush=True)
    rng = np.random.default_rng(0); si = 0
    for _ in range(nazar):
        si += mezcla(rng.permutation(p).tolist(), p, cl, S, cmax) is not None
    print(f"  azar: {si} de {nazar} permutaciones son mezcla de dos toroidales")
