"""
Mezclas de TRES toroidales (para las que no son mezcla de dos), con modelo nulo.
Por cada tope de clase c de las piezas: ¿hay A, B, C de clase <= c que entre las tres cubran T?
Uso: ../venv/bin/python mezclas3.py p [n_azar]
"""
import sys, json
import numpy as np
from mezclas import carga, superconjuntos


def analiza(T, p, cl, S, cmax):
    full = (1 << p) - 1
    pesos = (1 << np.arange(p, dtype=np.int64))
    m = ((S == np.asarray(T)) * pesos).sum(axis=1)
    out = {}
    for c in range(1, cmax + 1):
        sel = (m != full) & (cl <= c)
        has = np.zeros(1 << p, dtype=bool); has[m[sel]] = True
        sup = superconjuntos(has, p)
        ms = np.unique(m[sel])
        # mascaras maximales
        maxi = np.ones(len(ms), dtype=bool)
        for b in range(p):
            falta = (ms >> b) & 1 == 0
            maxi[falta] &= ~sup[ms[falta] | (1 << b)]
        mx = ms[maxi]
        dos = bool(sup[full ^ mx].any())
        tres = dos
        ej = None
        if not dos:
            for a in mx:
                resto = full ^ int(a)
                hit = sup[resto & ~mx]
                if hit.any():
                    tres = True; b = int(mx[np.argmax(hit)]); ej = (int(a), b, resto & ~b); break
        out[c] = (len(mx), dos, tres, ej)
    return out


if __name__ == "__main__":
    p = int(sys.argv[1]); nazar = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    cl, S = carga(p); cmax = int(cl.max())
    O = sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"]))
    for i, o in enumerate(O):
        r = analiza(o["rep"], p, cl, S, cmax)
        c2 = next((c for c in r if r[c][1]), None); c3 = next((c for c in r if r[c][2]), None)
        ej = r[c3][3] if c3 else None
        tam = [bin(x).count("1") for x in ej] if ej else ""
        print(f"  {i}:c{o['clase']}x{o['tam']:<4d} dos piezas: clase<={c2}   tres piezas: clase<={c3} {tam}", flush=True)
    rng = np.random.default_rng(0); n2 = [0] * (cmax + 1); n3 = [0] * (cmax + 1)
    for _ in range(nazar):
        r = analiza(rng.permutation(p).tolist(), p, cl, S, cmax)
        for c in r: n2[c] += r[c][1]; n3[c] += r[c][2]
    print(f"  azar ({nazar} permutaciones), por tope de clase 1..{cmax}: dos piezas {n2[1:]}, tres piezas {n3[1:]}")
