"""
Numero MINIMO de ondas planas para "levitar" cada solucion de las N reinas.

Modelo (ver README):
  - Tablero = red de trampas N x N, casillas x = (i, j), i = fila, j = columna.
  - Campo = suma de K ondas planas reales sobre esa red:
        f(x) = sum_k  A_k cos(2*pi*(k . x)/N + phi_k)
    con k = (u, v) en Z_N^2 (frecuencias conmensuradas con la red).
    La constante (k = 0) es un umbral libre y no cuenta como onda.
  - Una solucion "se levita" con esas K ondas si las N reinas son EXACTAMENTE
    las N casillas mas brillantes: f(reina) >= t + delta, f(resto) <= t - delta,
    con |a_k|, |b_k| <= 1 (a = A cos phi, b = -A sin phi) y margen relativo delta.

Minimo exacto de K: MILP (binaria z_k = "uso la onda k"), minimizar sum z_k.
Despues, con el soporte fijo, un LP maximiza el margen para dar parametros robustos.

Ademas se comprueba el caso "1 onda LIBRE" (k real, no en la rejilla): una sola
onda solo pinta un arco de fases, asi que basta ver si las reinas forman un arco
contiguo en el circulo de fases. Eso recoge la familia sturmiana.

Uso:  python min_waves.py 4 5 6 7 8 9 10 [--delta 1e-3]
Salida: levitation_queens/data/waves_N{N}.json  +  resumen en pantalla.
"""
import argparse, json, math, os, time
import numpy as np
from scipy.optimize import milp, linprog, LinearConstraint, Bounds

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- soluciones
def solutions(N):
    out = []
    def rec(r, c, d1, d2, p):
        if r == N:
            out.append(tuple(p)); return
        for x in range(N):
            if x in c or r + x in d1 or r - x in d2:
                continue
            p.append(x); rec(r + 1, c | {x}, d1 | {r + x}, d2 | {r - x}, p); p.pop()
    rec(0, set(), set(), set(), [])
    return out


# las 8 simetrias del tablero como mapas afines (i,j) -> L(i,j) + c  (sobre enteros)
def board_maps(N):
    m = N - 1
    return [
        ((1, 0, 0, 1), (0, 0)),   # identidad
        ((0, 1, -1, 0), (0, m)),  # rot 90:  (i,j) -> (j, m-i)
        ((-1, 0, 0, -1), (m, m)), # rot 180
        ((0, -1, 1, 0), (m, 0)),  # rot 270: (i,j) -> (m-j, i)
        ((1, 0, 0, -1), (0, m)),  # espejo columnas
        ((-1, 0, 0, 1), (m, 0)),  # espejo filas
        ((0, 1, 1, 0), (0, 0)),   # transpuesta
        ((0, -1, -1, 0), (m, m)), # antitranspuesta
    ]


def apply_map(g, perm):
    (a, b, c, d), (e, f) = g
    N = len(perm)
    out = [None] * N
    for i, j in enumerate(perm):
        i2, j2 = a * i + b * j + e, c * i + d * j + f
        out[i2] = j2
    return tuple(out)


def classes(sols, N):
    """rep -> lista de (solucion, g) con g(rep) = solucion."""
    G = board_maps(N)
    seen, reps = {}, {}
    for s in sols:
        if s in seen:
            continue
        reps[s] = []
        for g in G:
            t = apply_map(g, s)
            if t not in seen:
                seen[t] = s
                reps[s].append((t, g))
    return reps


# ---------------------------------------------------------------- frecuencias
def freq_pairs(N, fisico=False, nodeg=False):
    """Un representante por par {k, -k}, sin k = 0.
    fisico=True: solo ondas con longitud de onda en el plano >= 2 casillas
    (|k| <= N/2 usando el alias mas corto). Si la red de trampas se hace con la
    misma frecuencia, la separacion entre trampas es lambda0/2, y ninguna onda de
    esa frecuencia puede tener franjas mas juntas que lambda0 = 2 casillas."""
    ks = []
    for u in range(N):
        for v in range(N):
            if (u, v) == (0, 0):
                continue
            neg = ((-u) % N, (-v) % N)
            if neg < (u, v):
                continue
            if nodeg and 0 in (u % N, v % N, (u + v) % N, (u - v) % N):
                continue                     # la onda seria constante a lo largo de una linea de ataque
            if fisico:
                uu = u if u <= N // 2 else u - N
                vv = v if v <= N // 2 else v - N
                if uu * uu + vv * vv > (N / 2) ** 2 + 1e-9:
                    continue
            ks.append((u, v))
    return ks


def design(N, ks):
    """Columnas cos/sin de cada onda evaluadas en las N^2 casillas."""
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    I, J = I.ravel(), J.ravel()
    cols, owner, kind = [], [], []
    for n, (u, v) in enumerate(ks):
        th = 2 * np.pi * (u * I + v * J) / N
        cols.append(np.cos(th)); owner.append(n); kind.append('c')
        s = np.sin(th)
        if np.max(np.abs(s)) > 1e-9:          # k autoconjugada -> sin identicamente 0
            cols.append(s); owner.append(n); kind.append('s')
    return np.array(cols).T, np.array(owner), kind


def signs(perm, N):
    y = -np.ones(N * N)
    for i, j in enumerate(perm):
        y[i * N + j] = 1.0
    return y


# ---------------------------------------------------------------- MILP exacto
def min_support(perm, N, ks, X, owner, delta, time_limit):
    y = signs(perm, N)
    P, K = X.shape[1], len(ks)
    # variables: [coefs (P) | t | z (K)]
    nv = P + 1 + K
    # separacion: y * (X c - t) >= delta
    A1 = np.hstack([y[:, None] * X, -y[:, None], np.zeros((N * N, K))])
    cons = [LinearConstraint(A1, delta, np.inf)]
    # |c_p| <= z_owner(p)
    E = np.zeros((P, K)); E[np.arange(P), owner] = 1
    Ic = np.eye(P)
    cons.append(LinearConstraint(np.hstack([Ic, np.zeros((P, 1)), -E]), -np.inf, 0))
    cons.append(LinearConstraint(np.hstack([-Ic, np.zeros((P, 1)), -E]), -np.inf, 0))
    obj = np.r_[np.zeros(P + 1), np.ones(K)]
    lb = np.r_[-np.ones(P), -np.inf, np.zeros(K)]
    ub = np.r_[np.ones(P), np.inf, np.ones(K)]
    integ = np.r_[np.zeros(P + 1), np.ones(K)]
    r = milp(obj, constraints=cons, bounds=Bounds(lb, ub), integrality=integ,
             options={'time_limit': time_limit, 'mip_rel_gap': 0})
    if r.x is None:
        return [], r.status == 2, None       # status 2 = infactible (probado)
    z = r.x[P + 1:]
    sup = [n for n in range(K) if z[n] > 0.5]
    exact = r.status == 0
    lower = math.ceil(r.mip_dual_bound - 1e-6) if hasattr(r, 'mip_dual_bound') and r.mip_dual_bound is not None else None
    return sup, exact, lower


def best_params(perm, N, sup_ks):
    """Con el soporte fijo: maximiza el margen con |a|,|b| <= 1."""
    X, owner, kind = design(N, sup_ks)
    y = signs(perm, N)
    P = X.shape[1]
    # variables [c (P) | t | m]; max m  s.a.  y (X c - t) >= m
    A = np.hstack([-(y[:, None] * X), y[:, None], np.ones((N * N, 1))])
    r = linprog(np.r_[np.zeros(P + 1), -1.0], A_ub=A, b_ub=np.zeros(N * N),
                bounds=[(-1, 1)] * P + [(None, None), (0, None)], method='highs')
    c, t, m = r.x[:P], r.x[P], r.x[P + 1]
    waves = []
    for n, (u, v) in enumerate(sup_ks):
        a = c[(owner == n) & (np.array(kind) == 'c')].sum()
        b = c[(owner == n) & (np.array(kind) == 's')].sum()
        A_ = math.hypot(a, b)
        phi = math.atan2(-b, a)             # a cos th + b sin th = A cos(th + phi)
        uu = u if u <= N // 2 else u - N    # frecuencia con signo, mas corta
        vv = v if v <= N // 2 else v - N
        kn = math.hypot(uu, vv)
        waves.append({
            'k': [uu, vv],                              # ciclos a lo largo de (filas, columnas) en N casillas
            'amplitud': round(A_, 6),
            'fase_rad': round(phi, 6),
            'longitud_onda_casillas': round(N / kn, 6),
            'direccion_grados': round(math.degrees(math.atan2(uu, vv)), 3),  # 0 = a lo largo de las columnas
        })
    order = sorted(range(len(waves)), key=lambda n: -waves[n]['amplitud'])
    waves = [waves[n] for n in order]
    return waves, float(t), float(m)


def field(N, waves, t=0.0):
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    f = np.zeros((N, N))
    for w in waves:
        u, v = w['k']
        f += w['amplitud'] * np.cos(2 * np.pi * (u * I + v * J) / N + w['fase_rad'])
    return f - t


def verify(perm, N, waves, t):
    f = field(N, waves, t)
    top = set(np.argsort(f.ravel())[-N:])
    return top == {i * N + j for i, j in enumerate(perm)}


# ---------------------------------------------------------------- 1 onda libre
def one_free_wave(perm, N, D=8, fisico=False):
    """Busca k real = (u, v)/D (rejilla fina) tal que las reinas ocupen un arco
    contiguo de fases. Devuelve k o None. Es una busqueda en rejilla: None no
    prueba imposibilidad, solo que no hay k con denominador <= D*N que sirva."""
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    I, J = I.ravel(), J.ravel()
    q = np.zeros(N * N, bool)
    for i, j in enumerate(perm):
        q[i * N + j] = True
    M = N * D
    for uu in range(0, M):
        for vv in range(-M + 1, M):
            if uu == 0 and vv <= 0:
                continue
            if fisico:                             # alias mas corto de cada componente
                a1 = uu / D if uu / D <= N / 2 else uu / D - N
                a2 = vv / D - N * round(vv / D / N)
                if a1 * a1 + a2 * a2 > (N / 2) ** 2 + 1e-9:
                    continue
            th = ((uu * I + vv * J) % M) / M       # fase en [0,1)
            o = np.argsort(th, kind='stable')
            s, qs = th[o], q[o]
            if len(np.unique(np.round(s, 12))) < len(s) and _tie_mixes(s, qs):
                continue
            # reinas contiguas en el circulo?
            idx = np.flatnonzero(qs)
            ext = np.r_[qs, qs]
            for st in idx:
                if ext[st:st + N].all():
                    k1 = uu / D if uu / D <= N / 2 else uu / D - N
                    k2 = vv / D - N * round(vv / D / N)
                    return [k1, k2]
    return None


def _tie_mixes(s, qs):
    r = np.round(s, 12)
    for val in np.unique(r):
        g = qs[r == val]
        if g.any() and not g.all():
            return True
    return False


# ---------------------------------------------------------------- main
def _solve_class(args):
    rep, N, delta, time_limit, D, fisico = args[:6]
    nodeg = args[6] if len(args) > 6 else False
    ks = freq_pairs(N, fisico, nodeg)
    X, owner, _ = design(N, ks)
    sup, exact, lower = min_support(rep, N, ks, X, owner, delta, time_limit)
    free = one_free_wave(rep, N, D, fisico) if len(sup) != 1 else 'rejilla'
    return sup, exact, lower, free


def _solve_class_ckpt(args):
    cid, job = args
    t = time.time()
    sup, exact, lower, free = _solve_class(job)
    return cid, sup, exact, lower, free, time.time() - t


def _free_k_for(free, g, waves):
    """k de la onda libre para un miembro de la clase (k' = L k)."""
    if free == 'rejilla':
        return waves[0]['k']
    if free is None:
        return None
    (a, b, c, d), _ = g
    u, v = free
    return [a * u + b * v, c * u + d * v]


def run(N, delta, time_limit, D, workers, fisico=False, nodeg=False, muestra=None, seed=0):
    from multiprocessing import Pool
    t0 = time.time()
    sols = solutions(N)
    reps = classes(sols, N)
    if muestra:                               # solo una muestra aleatoria de clases (N grandes)
        import random
        allreps = list(reps.items())
        keep = sorted(random.Random(seed).sample(range(len(allreps)), min(muestra, len(allreps))))
        reps = dict(allreps[c] for c in keep)
        sols = [s for rep in reps for s, _ in reps[rep]]
    ks = freq_pairs(N, fisico, nodeg)
    idx = {s: n for n, s in enumerate(sols)}
    recs = [None] * len(sols)
    jobs = [(rep, N, delta, time_limit, D, fisico, nodeg) for rep in reps]
    # --- punto de control: una linea JSON por clase terminada; al relanzar se salta lo hecho
    tag = f'waves_N{N}' + ('_fisico' if fisico else '') + ('_nodeg' if nodeg else '') + (f'_muestra{muestra}s{seed}' if muestra else '')
    ckdir = os.path.join(HERE, 'data', 'checkpoints')
    os.makedirs(ckdir, exist_ok=True)
    ckpath = os.path.join(ckdir, tag + '.jsonl')
    done = {}
    if os.path.exists(ckpath):
        with open(ckpath) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:      # linea cortada por una parada brusca
                    continue
                if d['n_clases'] == len(reps) and d['nodeg'] == nodeg and d['fisico'] == fisico:
                    done[d['clase']] = d
    pend = [(cid, j) for cid, j in enumerate(jobs) if cid not in done]
    print(f"N={N}: {len(reps)} clases, {len(done)} ya hechas (checkpoint), {len(pend)} pendientes -> {ckpath}", flush=True)
    if pend:
        with Pool(workers) as pool, open(ckpath, 'a') as fh:
            for n, (cid, sup, exact, lower, free, secs) in enumerate(
                    pool.imap_unordered(_solve_class_ckpt, pend, chunksize=1), 1):
                d = {'clase': cid, 'n_clases': len(reps), 'nodeg': nodeg, 'fisico': fisico,
                     'sup_k': [list(ks[i]) for i in sup], 'exacto': bool(exact), 'cota_inferior': lower,
                     'libre': free, 'seg': round(secs, 1), 'delta': delta, 'time_limit': time_limit}
                fh.write(json.dumps(d) + '\n'); fh.flush(); os.fsync(fh.fileno())
                done[cid] = d
                print(f"  [{len(done)}/{len(reps)}] clase {cid}: {len(sup)} ondas"
                      f"{'' if exact else ' (cota)'}  {secs:.0f}s  | total {time.time()-t0:.0f}s", flush=True)
    kidx = {tuple(k): i for i, k in enumerate(ks)}
    res = [([kidx[tuple(k)] for k in done[c]['sup_k']], done[c]['exacto'], done[c]['cota_inferior'],
            done[c]['libre']) for c in range(len(reps))]
    for cid, ((rep, members), (sup, exact, lower, free)) in enumerate(zip(reps.items(), res)):
        for sol, g in members:
            if not sup:                      # sin solucion factible en este conjunto de ondas
                recs[idx[sol]] = {'id': idx[sol], 'solucion': list(sol), 'clase': cid,
                                  'ondas_min': None, 'exacto': bool(exact), 'cota_inferior': lower,
                                  'ondas': [], 'una_onda_libre': _free_k_for(free, g, [])}
                continue
            (a, b, c, d), _ = g
            # g^-1 tiene parte lineal L^-1 = L^T (ortogonal); k' = L^-T k = L k
            sup_k = []
            for n in sup:
                u, v = ks[n]
                sup_k.append(((a * u + b * v) % N, (c * u + d * v) % N))
            waves, t, m = best_params(sol, N, sup_k)
            assert verify(sol, N, waves, t), (N, sol)
            recs[idx[sol]] = {
                'id': idx[sol], 'solucion': list(sol), 'clase': cid,
                'ondas_min': len(sup), 'exacto': bool(exact),
                'cota_inferior': lower if not exact else len(sup),
                'umbral': round(t, 6), 'margen': round(m, 6),
                'ondas': waves,
                'una_onda_libre': _free_k_for(free, g, waves),   # k de UNA onda (quizas no entera) que basta, o null
            }
    out = {'N': N, 'delta': delta, 'fisico': fisico, 'nodeg': nodeg, 'n_soluciones': len(sols), 'n_clases': len(reps),
           'modelo': 'f(i,j) = sum A cos(2pi(k0*i + k1*j)/N + fase); reinas = N casillas con f > umbral',
           'soluciones': recs}
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    if muestra:
        out['muestra'] = {'clases': len(reps), 'semilla': seed, 'nota': 'clase = indice dentro de la muestra'}
    with open(os.path.join(HERE, 'data', tag + '.json'), 'w') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    from collections import Counter
    h = Counter(r['ondas_min'] or 0 for r in recs)
    ne = sum(not r['exacto'] for r in recs)
    fr = sum(r['una_onda_libre'] is not None for r in recs)
    print(f"{'[fisico] ' if fisico else ''}{'[nodeg] ' if nodeg else ''}N={N}: {len(sols)} sol, {len(reps)} clases | ondas_min {dict(sorted(h.items()))}"
          f" | no exactas {ne} | 1 onda libre {fr} | {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Ns', type=int, nargs='+')
    ap.add_argument('--delta', type=float, default=1e-3)
    ap.add_argument('--time-limit', type=float, default=120)
    ap.add_argument('--D', type=int, default=8)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--fisico', action='store_true', help='solo ondas con lambda >= 2 casillas')
    ap.add_argument('--nodeg', action='store_true', help='sin ondas con proyeccion 0 sobre fila/col/diagonal')
    ap.add_argument('--muestra', type=int, help='solo esta cantidad de clases al azar')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    for N in a.Ns:
        run(N, a.delta, a.time_limit, a.D, a.workers, a.fisico, a.nodeg, a.muestra, a.seed)
