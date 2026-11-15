"""
Combinaciones lineales de piezas en el cilindro de p:  T = l*X + m*Y  (modulo p)  y  T = X + Y - Z.

Piezas (funciones fila -> columna, salvo sumar una constante = girar el cilindro):
  H   helices de paso real: floor(s*r + c) (helices.py)
  Q   toroidales de q (por defecto 13) metidas en el cilindro de p: angulo estirado
      floor(p*U/q + c) y filas repetidas (r mod q) o estiradas (floor(q*r/p)).
Se cuenta, para cada solucion de p y para permutaciones al azar, de cuantas formas sale.

Uso: ../venv/bin/python combina.py p q [n_azar]
"""
import sys, json
import numpy as np
from collections import Counter
from helices import helices

W = None
def h(X): return (X.astype(np.uint64) * W[:X.shape[-1]]).sum(axis=-1)
def norm(X, p): return (X - X[..., :1]) % p


def piezas_q(p, q):
    U = np.loadtxt(f"data/clases_toro_p{q}.txt", dtype=np.int64)[:, 1:]
    r = np.arange(p); out = []
    for filas in (r % q, (q * r) // p):
        V = U[:, filas]                                        # q -> p filas
        for k in range(q):
            out.append((p * V + k * p // q) // q)              # angulo estirado, varios arranques
    return np.unique(norm(np.concatenate(out) % p, p), axis=0)


if __name__ == "__main__":
    p, q = int(sys.argv[1]), int(sys.argv[2]); nazar = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    W = np.random.default_rng(3).integers(1, 2**62, size=64, dtype=np.int64).astype(np.uint64)
    H = np.unique(norm(helices(p), p), axis=0); Q = piezas_q(p, q)
    P = {"H": H, "Q": Q, "H+Q": np.unique(np.concatenate([H, Q]), axis=0)}
    print(f"p={p}: {len(H)} helices, {len(Q)} piezas de q={q}")
    # conjunto de m*Y normalizados, para cada familia
    mult = {k: np.sort(np.unique(np.concatenate([h(norm(m * V % p, p)) for m in range(1, p)]))) for k, V in P.items()}
    hH = np.sort(h(H))
    def esta(hs, S):
        i = np.searchsorted(S, hs); i[i == len(S)] = 0
        return S[i] == hs
    def mide(T):
        T = np.asarray(T); res = {}
        for k, V in P.items():
            n = 0
            for l in range(1, p):
                n += int(esta(h(norm((T - l * V) % p, p)), mult[k]).sum())
            res[f"lX+mY ({k})"] = n
        n = 0                                                   # X + Y - Z con helices
        for X in H:
            n += int(esta(h(norm((T - X - H) % p, p)) , np.sort(h(norm((-H) % p, p)))).sum())
        res["X+Y-Z (H)"] = n
        return res
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    from ordenadas import Orbitas
    Ob = Orbitas(p); expl = {Ob.clave(g["rep"]) for g in json.load(open(f"data/ordenadas_p{p}.json"))}
    tot = {}
    for i, o in enumerate(O):
        g = "explicada" if Ob.clave(o["rep"]) in expl else "SIN EXPLICAR"
        res = mide(o["rep"]); tot.setdefault(g, []).append(res)
        print(f"  {i}:c{o['clase']}x{o['tam']:<5d} {g:12s} {res}", flush=True)
    rng = np.random.default_rng(0)
    for _ in range(nazar): tot.setdefault("azar", []).append(mide(rng.permutation(p).tolist()))
    for g, L in tot.items():
        print(f"  {g:12s} " + "  ".join(f"{k}: mediana {int(np.median([x[k] for x in L]))}, con alguna {sum(x[k] > 0 for x in L)}/{len(L)}" for k in L[0]))
