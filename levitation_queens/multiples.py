"""
Multiplos: ¿las ondas de dos soluciones pequenas levitan su producto?

Producto (fractal_queens/tensor.py): rho(i*m + j) = sigma(i)*m + tau(j), valido
para toda sigma de n reinas si tau (m reinas) es toroidal. N = n*m.

Ondas candidatas en el tablero N x N (k en ciclos por tablero N):
  gruesas  : las ondas de sigma con el MISMO k  (dan el patron de bloques)
  finas    : la onda de tau multiplicada por n  (repite tau en cada bloque)
  cruzadas : k_sigma ± n*k_tau                  (los terminos del producto cos*cos)

Para cada producto se prueba con un LP (solo amplitudes, fases y umbral):
  G+F   = gruesas + finas
  G+F+C = ademas las cruzadas
y se cuenta cuantas ondas quedan con amplitud no nula.

Uso: python multiples.py
"""
import json, os
from collections import Counter
import numpy as np
from transfer import lp_margin, params, DELTA
import min_waves as mw


def load(N):
    return json.load(open(os.path.join(mw.HERE, 'data', f'waves_N{N}.json')))['soluciones']


def product(sig, tau):
    n, m = len(sig), len(tau)
    return tuple(sig[i] * m + tau[j] for i in range(n) for j in range(m))


def test(n, m):
    S, T = load(n), [r for r in load(m) if r['ondas_min'] == 1]      # tau toroidal = 1 onda
    res = Counter()
    ejemplos = {}
    for rs in S:
        for rt in T:
            rho = product(rs['solucion'], rt['solucion'])
            N = n * m
            assert mw.verify.__code__ and len(set(rho)) == N
            G = [tuple(w['k']) for w in rs['ondas']]
            F = [tuple(n * x for x in w['k']) for w in rt['ondas']]
            C = [(g[0] + s * f[0], g[1] + s * f[1]) for g in G for f in F for s in (1, -1)]
            out = []
            for name, ks in (('G+F', G + F), ('G+F+C', G + F + C)):
                mg, x = lp_margin(rho, N, ks)
                if mg >= DELTA:
                    waves, t = params(N, ks, x)
                    ok = mw.verify(rho, N, waves, t)
                    out.append((name, ok, len(waves), round(mg, 4)))
                    if ok and name not in ejemplos:
                        ejemplos[name] = {'N': N, 'sigma': rs['solucion'], 'tau': rt['solucion'],
                                          'solucion': list(rho), 'umbral': t, 'margen': mg, 'ondas': waves}
                else:
                    out.append((name, False, None, round(mg, 4)))
            key = tuple((o[0], o[1], o[2]) for o in out)
            res[key] += 1
    return res, ejemplos, len(S), len(T)


if __name__ == '__main__':
    allex = {}
    for n, m in ((4, 5), (5, 5), (6, 5), (7, 5), (4, 7), (5, 7), (8, 5)):
        res, ex, ns, nt = test(n, m)
        print(f"N = {n}x{m} = {n*m}:  {ns} sigma x {nt} tau toroidales = {ns*nt} productos")
        for k, c in sorted(res.items(), key=lambda x: -x[1]):
            print(f"   {c:4d}  " + "   ".join(f"{a}: {'SI' if b else 'no'}" + (f" ({w} ondas)" if b else '') for a, b, w in k))
        allex[f'{n}x{m}'] = ex
    json.dump(allex, open(os.path.join(mw.HERE, 'data', 'multiples_ejemplos.json'), 'w'), indent=1)
