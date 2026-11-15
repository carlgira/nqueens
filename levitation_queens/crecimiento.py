"""
Funcion de crecimiento de los niveles desde las toroidales.

Hipotesis:  nuevas(n+1) = beta * H(n)^2 * (1 - H(n)/Q)      H = soluciones acumuladas
(cada par de soluciones conocidas tiene una probabilidad fija de fusionarse en algo nuevo).
Sin el freno final, 1/H baja beta por nivel  ->  el "desierto" dura ~ 1/(beta*H0) niveles.

Uso: python crecimiento.py            (N=11, 13 con los niveles guardados)
     python crecimiento.py --correr 17 --hasta 6000     (calcula niveles reales de p=17)
"""
import argparse, os, time
import numpy as np
from perfil import Q

HERE = os.path.dirname(os.path.abspath(__file__))


def table(N, cum):
    print(f"===== N={N}  (Q = {Q[N]})")
    print("  nivel |      H | nuevas | nuevas/H^2 (x1e-4) | con freno (x1e-4)")
    b = []
    for n in range(len(cum) - 1):
        H, new = cum[n], cum[n + 1] - cum[n]
        r = new / H ** 2; rf = r / (1 - H / Q[N])
        b.append(rf)
        print(f"  {n:5d} | {H:6d} | {new:6d} | {1e4*r:10.2f} | {1e4*rf:10.2f}")
    return b


def simulate(H0, beta, Qn, maxlev=200):
    seq = [H0]
    while len(seq) < maxlev:
        H = seq[-1]
        new = beta * H * H * (1 - H / Qn)
        if new < 0.5 or H + new > Qn:
            seq.append(min(Qn, H + new)); break
        seq.append(H + new)
    return seq


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--correr', type=int, default=0)
    ap.add_argument('--hasta', type=int, default=6000)
    a = ap.parse_args()
    if a.correr:
        from fusion import wave_gates
        from fusion_ciega import close_sym, fuse_halves
        p = a.correr
        have = {tuple((s * i + c) % p for i in range(p)) for s in range(2, p - 1) for c in range(p)}
        gates = list(wave_gates(p)); cum = [len(have)]; t0 = time.time()
        while len(have) < a.hasta:
            new, _, _ = fuse_halves(have, p, gates)
            new = close_sym(new, p) - have
            if not new:
                break
            have |= new; cum.append(len(have))
            print(f"   nivel {len(cum)-1}: {len(have)} ({time.time()-t0:.0f}s)", flush=True)
            np.save(os.path.join(HERE, 'data', f'crecimiento_p{p}.npy'), np.array(cum))
        table(p, cum)
    else:
        for N in (11, 13):
            lev = np.load(os.path.join(HERE, 'data', f'niveles_N{N}_tramo.npy'))
            cum = [int(x) for x in np.cumsum([int((lev == L).sum()) for L in range(lev.max() + 1)])]
            b = table(N, cum)
            if N == 13:
                beta = float(np.median(b[4:17]))
                sim = simulate(cum[4], beta, Q[N])
                print(f"  beta (mediana niveles 4-16) = {beta:.3e};  1/(beta*H) en el nivel 4 = {1/(beta*cum[4]):.1f} niveles")
                print("  real     : " + " ".join(str(x) for x in cum[4:]))
                print("  simulado : " + " ".join(str(int(round(x))) for x in sim))
