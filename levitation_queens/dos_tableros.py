"""
¿Una solucion es la INTERSECCION o el XOR de dos tableros generados con ondas?

Tablero b (b = 1, 2):  B_b = { casillas x : f_b(x) > t_b },  f_b = suma de pocas ondas planas
(cualquier numero de casillas encendidas).

  interseccion:  reinas = B_1 y B_2      (toda reina encendida en los dos; cada casilla vacia
                                           apagada en al menos uno)
  xor:           reinas = B_1 xor B_2    (toda reina en exactamente uno; cada vacia en los dos
                                           o en ninguno)

MILP exacto: minimizar el numero TOTAL de ondas (las de un tablero + las del otro), con una
binaria por onda y tablero y una binaria por casilla (en que tablero esta encendida).
Se compara con las ondas minimas del metodo directo (un solo campo).

Uso: python dos_tableros.py 7 8 [--modo and xor] [--kmax 3] [--time-limit 120]
Salida: data/dos_tableros_N{N}.json
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))
DELTA = 1e-3


def solve(perm, N, mode, kmax, tl):
    ks = mw.freq_pairs(N, False, True)
    X, owner, _ = mw.design(N, ks)
    P, K, C = X.shape[1], len(ks), N * N
    q = mw.signs(perm, N) > 0
    M = 2.0 * (2 * kmax + 1) + 1            # |f - t| <= 2*kmax (coef en [-1,1], 2 por onda) + |t|
    # variables: [c1 (P) | t1 | c2 (P) | t2 | z1 (K) | z2 (K) | b1 (C) | b2 (C)]
    o_c1, o_t1, o_c2, o_t2 = 0, P, P + 1, 2 * P + 1
    o_z1, o_z2 = 2 * P + 2, 2 * P + 2 + K
    o_b1, o_b2 = 2 * P + 2 + 2 * K, 2 * P + 2 + 2 * K + C
    nv = o_b2 + C
    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        r = np.zeros(nv)
        for idx, v in coefs:
            r[idx] = v
        rows.append(r); lo.append(l); hi.append(h)

    for b, (oc, ot, ob) in enumerate(((o_c1, o_t1, o_b1), (o_c2, o_t2, o_b2))):
        for x in range(C):
            base = [(oc + p, X[x, p]) for p in range(P)] + [(ot, -1.0)]
            # encendida (b=1):  f - t >= DELTA      ->  f - t - M*b >= DELTA - M
            add(base + [(ob + x, -M)], DELTA - M, np.inf)
            # apagada  (b=0):  f - t <= -DELTA      ->  f - t - M*b <= -DELTA
            add(base + [(ob + x, -M)], -np.inf, -DELTA)
    for x in range(C):
        if mode == 'and':
            if q[x]:
                add([(o_b1 + x, 1)], 1, 1); add([(o_b2 + x, 1)], 1, 1)
            else:
                add([(o_b1 + x, 1), (o_b2 + x, 1)], -np.inf, 1)
        else:                                   # xor
            if q[x]:
                add([(o_b1 + x, 1), (o_b2 + x, 1)], 1, 1)
            else:
                add([(o_b1 + x, 1), (o_b2 + x, -1)], 0, 0)
    E = np.zeros((P, K)); E[np.arange(P), owner] = 1
    for oc, oz in ((o_c1, o_z1), (o_c2, o_z2)):
        for p in range(P):
            add([(oc + p, 1), (oz + owner[p], -1)], -np.inf, 0)
            add([(oc + p, -1), (oz + owner[p], -1)], -np.inf, 0)
        add([(oz + k, 1) for k in range(K)], 0, kmax)
    # romper la simetria entre tableros: el 1 usa al menos tantas ondas como el 2
    add([(o_z1 + k, 1) for k in range(K)] + [(o_z2 + k, -1) for k in range(K)], 0, np.inf)
    obj = np.zeros(nv); obj[o_z1:o_z2 + K] = 1
    lb = -np.ones(nv); ub = np.ones(nv)
    lb[o_t1] = lb[o_t2] = -2 * kmax; ub[o_t1] = ub[o_t2] = 2 * kmax
    lb[o_z1:] = 0
    integ = np.zeros(nv); integ[o_z1:] = 1
    t0 = time.time()
    r = milp(obj, constraints=LinearConstraint(np.array(rows), lo, hi), bounds=Bounds(lb, ub),
             integrality=integ, options={'time_limit': tl, 'mip_rel_gap': 0})
    if r.x is None:
        return {'ondas': None, 'estado': 'infactible' if r.status == 2 else 'tiempo', 'seg': round(time.time() - t0)}
    z1 = [ks[k] for k in range(K) if r.x[o_z1 + k] > .5]
    z2 = [ks[k] for k in range(K) if r.x[o_z2 + k] > .5]
    b1 = r.x[o_b1:o_b1 + C] > .5
    b2 = r.x[o_b2:o_b2 + C] > .5
    res = (b1 & b2) if mode == 'and' else (b1 ^ b2)
    assert (res == q).all()
    return {'ondas': len(z1) + len(z2), 'k1': [list(k) for k in z1], 'k2': [list(k) for k in z2],
            'encendidas': [int(b1.sum()), int(b2.sum())], 'exacto': r.status == 0,
            'cota_inf': int(np.ceil(r.mip_dual_bound - 1e-6)) if r.status != 0 else len(z1) + len(z2),
            'seg': round(time.time() - t0)}


def _job(args):
    cid, perm, N, mode, kmax, tl = args
    return cid, mode, solve(perm, N, mode, kmax, tl)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--modo', nargs='+', default=['and', 'xor'])
    ap.add_argument('--kmax', type=int, default=3)
    ap.add_argument('--time-limit', type=float, default=120)
    ap.add_argument('--max-clases', type=int, default=0)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    for N in a.Ns:
        W = json.load(open(os.path.join(HERE, 'data', f'waves_N{N}_nodeg.json')))['soluciones']
        reps = {}
        for r in W:
            reps.setdefault(r['clase'], r)
        cl = sorted(reps)
        if a.max_clases and len(cl) > a.max_clases:
            import random
            cl = sorted(random.Random(0).sample(cl, a.max_clases))
        jobs = [(c, tuple(reps[c]['solucion']), N, m, a.kmax, a.time_limit) for c in cl for m in a.modo]
        with Pool(a.workers) as p:
            out = p.map(_job, jobs, chunksize=1)
        tab = {}
        for cid, mode, res in out:
            tab.setdefault(cid, {})[mode] = res
        print(f"N={N}: clase | ondas directo | interseccion: total (k1+k2) [encendidas] | xor: total (k1+k2) [encendidas]")
        for c in cl:
            line = f"   {c:3d} {reps[c]['solucion']} | {reps[c]['ondas_min']} |"
            for m in a.modo:
                r = tab[c][m]
                if r['ondas'] is None:
                    line += f" {r['estado']} ({r['seg']}s) |"
                else:
                    line += (f" {r['ondas']}{'' if r['exacto'] else '*'} ({len(r['k1'])}+{len(r['k2'])}) {r['encendidas']} |")
            print(line, flush=True)
        json.dump({str(c): {'solucion': reps[c]['solucion'], 'directo': reps[c]['ondas_min'], **tab[c]} for c in cl},
                  open(os.path.join(HERE, 'data', f'dos_tableros_N{N}.json'), 'w'), indent=1)
