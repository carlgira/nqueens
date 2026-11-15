"""
¿Las toroidales sin descripcion son "tres menos una"?   T = A + B - C   (fila a fila, modulo p)

A y B: soluciones de una o dos espirales (todas sus traslaciones y simetrias).
C: (i) tambien de una o dos espirales, (ii) cualquier solucion ya explicada (ordenadas.py),
   (iii) cualquier toroidal.
La relacion se conserva al trasladar o escalar las cuatro, pero no con los giros: por eso se
prueban las 8 imagenes por giro de cada objetivo T. Se fija A(0) = 0 (sumar t a A y restarlo a B
da lo mismo).
Azar: el mismo intento con permutaciones al azar como objetivo.

Uso: ../venv/bin/python sumas.py p
"""
import sys, json
import numpy as np
from orbitas import mats, apply
from haces import canon


def todas_traslaciones(U, p):
    U = np.asarray(U); r = np.arange(p)
    return [(np.roll(U, -s) + t) % p for s in range(p) for t in range(p)]


def expande(reps, p, Ms):
    """todas las soluciones (giros, escalas, traslaciones) de una lista de representantes."""
    vistos = set(); out = []
    for T in reps:
        for M in Ms:
            k = canon([int(x) for x in apply(M, T, p)], p)
            if k in vistos: continue
            vistos.add(k); out.extend(todas_traslaciones(k, p))
    return np.unique(np.array(out, dtype=np.int64), axis=0)


W = None
def h(X):
    return (X.astype(np.uint64) * W[:X.shape[-1]]).sum(axis=-1)


def id_traslacion(X, p):
    """identificador de la orbita de traslacion de cada fila (minimo hash entre las p normalizaciones)."""
    X = np.atleast_2d(X); out = np.empty(len(X), dtype=np.uint64)
    for i in range(0, len(X), 20000):
        Y = X[i:i + 20000]
        best = None
        for s in range(p):
            Z = (np.roll(Y, -s, axis=1) - Y[:, s:s + 1]) % p
            hz = h(Z); best = hz if best is None else np.minimum(best, hz)
        out[i:i + 20000] = best
    return out


def representa(T, A0, base, conj, lab, mio):
    """etiquetas (orbita de C) de todas las (A, B, C) con T = A + B - C, quitando C de la orbita de T."""
    out = []
    for A in A0:
        hs = h((A + base - T) % p)
        i = np.searchsorted(conj, hs); i[i == len(conj)] = 0
        ok = conj[i] == hs
        out.extend(lab[i[ok]].tolist())
    return [x for x in out if x != mio]


if __name__ == "__main__":
    p = int(sys.argv[1])
    W = np.random.default_rng(7).integers(1, 2**62, size=64, dtype=np.int64).astype(np.uint64)
    Ms = mats(p); giros = []
    for M in Ms:                                    # una matriz por giro (u = 1)
        if not any(np.array_equal(np.array(M) % p, np.array(G) * u % p) for G in giros for u in range(1, p)):
            giros.append(M)
    assert len(giros) == 8
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    # orbita (del grupo completo) de cada orbita de traslacion
    orb = {}
    for i, o in enumerate(O):
        for M in Ms: orb[int(id_traslacion(np.asarray(apply(M, o["rep"], p)), p)[0])] = i
    G = json.load(open(f"data/ordenadas_p{p}.json"))
    gi = {orb[int(id_traslacion(np.asarray(g["rep"]), p)[0])]: g["familia"][:2] for g in G}
    tipo = {i: ("1-2 espirales" if gi.get(i) in ("F0", "F1") else "explicada" if i in gi else "SIN EXPLICAR") for i in range(len(O))}
    todas = np.loadtxt(f"data/clases_toro_p{p}.txt", dtype=np.int64)[:, 1:]
    lab_t = np.array([orb[int(x)] for x in id_traslacion(todas, p)])
    hs = h(todas); orden = np.argsort(hs); conj = hs[orden]; lab = lab_t[orden]
    base = todas[np.isin(lab_t, [i for i in tipo if tipo[i] == "1-2 espirales"])]
    A0 = base[base[:, 0] == 0]
    print(f"p={p}: A, B entre {len(base)} soluciones de 1-2 espirales; C cualquiera de las {len(todas)} toroidales"
          f" salvo las de la misma orbita que T")
    rng = np.random.default_rng(0)
    objetivos = [(f"{i}:c{o['clase']}x{o['tam']} ({tipo[i]})", o["rep"], i) for i, o in enumerate(O) if o["clase"] >= 3]
    objetivos += [(f"azar {k}", rng.permutation(p).tolist(), -1) for k in range(3)]
    for nom, T, i in objetivos:
        labs = []
        for M in giros:
            labs += representa(np.asarray(apply(M, T, p)), A0, base, conj, lab, i)
        from collections import Counter
        por = Counter(tipo[x] for x in labs)
        print(f"  {nom:34s} {len(labs):6d} formas;  C de tipo: {dict(por)};  C en orbitas {sorted(set(labs))}", flush=True)
