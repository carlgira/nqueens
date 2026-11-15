"""
Newton Queens -- estudio de cuencas de atraccion
================================================

Preguntas:

  A. De cuantas condiciones iniciales distintas se llega a UNA MISMA solucion
     final? (tamano de la cuenca de cada solucion). Es lo que el usuario
     pregunta: cuantas "formas iniciales" desembocan en el mismo atractor.

  B. Que aleatoriedad manda? Comparamos dos fuentes:
        B1: MISMA posicion inicial, energias (velocidades) distintas por reina.
        B2: MISMA energia (velocidades fijas), posiciones distintas.
     Para cada ancla contamos cuantas soluciones DISTINTAS alcanza.

  C. Ley del desplazamiento: la reina media acaba a ~N/2 casillas de donde
     empezo? (para N=8 medimos 3.8). Lo comprobamos por N.

  D. Hipotesis de los primos: para N coprimo con 6 existen soluciones
     MODULARES/toroidales (perm[i] = a*i + b mod N), las mas estructuradas.
     Tienen cuencas mas grandes? Correlaciona el tamano de cuenca con la
     estructura (modularidad, orden de simetria D4)?

Uso:
    python basin_study.py [N] [TRIALS] [MAX_STEPS] [V0]
    python basin_study.py 7 12000
    python basin_study.py 11 40000 11000 4.5
"""
from __future__ import annotations
import sys, time
import numpy as np
from collections import Counter

N         = int(sys.argv[1]) if len(sys.argv) > 1 else 7
TRIALS    = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6500
V0        = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
CHECK_EVERY, CHUNK, SEED = 5, 2000, 20260713

A, B, SIGMA = 2.5, 8.0, 0.5
GAMMA, K_SPREAD, MASS_COUPLE = 0.03, 0.7, 0.8
WALL_K, DT = 30.0, 0.012
TAU, S2 = 2 * np.pi, 2 * 0.5 ** 2


# ------------------------------ fisica vectorizada --------------------------
def accel(pos, vel, k, mass):
    x, y = pos[..., 0], pos[..., 1]
    fx = -TAU * A * np.sin(TAU * x); fy = -TAU * A * np.sin(TAU * y)
    dx = x[:, :, None] - x[:, None, :]; dy = y[:, :, None] - y[:, None, :]
    u, v = dx - dy, dx + dy
    T1 = np.exp(-(dx*dx)/S2); T2 = np.exp(-(dy*dy)/S2)
    T3 = np.exp(-(u*u)/S2);   T4 = np.exp(-(v*v)/S2)
    kj = k[:, None, :] / SIGMA**2
    fx += (kj*(dx*T1 + u*T3 + v*T4)).sum(2); fy += (kj*(dy*T2 - u*T3 + v*T4)).sum(2)
    lo, hi = 0.0, N-1.0
    fx += WALL_K*np.maximum(lo-x,0) - WALL_K*np.maximum(x-hi,0)
    fy += WALL_K*np.maximum(lo-y,0) - WALL_K*np.maximum(y-hi,0)
    fx -= GAMMA*vel[...,0]; fy -= GAMMA*vel[...,1]
    return np.stack([fx, fy], -1) / mass[..., None]


def rk4(pos, vel, k, mass):
    a1 = accel(pos, vel, k, mass)
    p2, v2 = pos+.5*DT*vel, vel+.5*DT*a1; a2 = accel(p2, v2, k, mass)
    p3, v3 = pos+.5*DT*v2, vel+.5*DT*a2;  a3 = accel(p3, v3, k, mass)
    p4, v4 = pos+DT*v3, vel+DT*a3;        a4 = accel(p4, v4, k, mass)
    dpos = (DT/6)*(vel+2*v2+2*v3+v4)
    return pos+dpos, vel+(DT/6)*(a1+2*a2+2*a3+a4), dpos


def cells_of(pos): return np.clip(np.rint(pos), 0, N-1).astype(np.int32)


def solved_of(pos):
    c = cells_of(pos); cx, cy = c[..., 0], c[..., 1]
    sc = cx[:,:,None]==cx[:,None,:]; sr = cy[:,:,None]==cy[:,None,:]
    sd = np.abs(cx[:,:,None]-cx[:,None,:])==np.abs(cy[:,:,None]-cy[:,None,:])
    at = (sc|sr|sd) & ~np.eye(N, dtype=bool)[None]
    return (at.any(2).sum(1) == 0), c


def run_batch(pos, vel, k, mass, track_path=False):
    bt = pos.shape[0]
    solved_step = np.full(bt, -1, np.int32)
    final = np.zeros((bt, N, 2), np.int32)
    pos0 = pos.copy(); pathlen = np.zeros((bt, N)); active = np.ones(bt, bool)
    for step in range(MAX_STEPS):
        pos, vel, dpos = rk4(pos, vel, k, mass)
        if track_path:
            pathlen += np.linalg.norm(dpos, axis=2) * active[:, None]
        if step % CHECK_EVERY == 0 or step == MAX_STEPS-1:
            sol, c = solved_of(pos)
            fresh = sol & active
            if fresh.any():
                solved_step[fresh] = step; final[fresh] = c[fresh]; active[fresh] = False
            if not active.any(): break
    disp = np.linalg.norm(cells_of(pos) - pos0, axis=2)
    return solved_step, final, disp, pathlen


def make_random(bt, rng):
    pos = rng.uniform(0, N-1.0, (bt, N, 2))
    vel = rng.normal(0, V0, (bt, N, 2))
    k = np.clip(B*(1+K_SPREAD*rng.uniform(-1, 1, (bt, N))), 0.05*B, None)
    mass = 1 - MASS_COUPLE + MASS_COUPLE*k/k.mean(1, keepdims=True)
    return pos, vel, k, mass


# ------------------------------ estructura ----------------------------------
def d4_maps():
    n = N-1
    return [(np.array([[1,0],[0,1]]), np.array([0,0])),
            (np.array([[-1,0],[0,1]]), np.array([n,0])),
            (np.array([[1,0],[0,-1]]), np.array([0,n])),
            (np.array([[-1,0],[0,-1]]), np.array([n,n])),
            (np.array([[0,1],[1,0]]), np.array([0,0])),
            (np.array([[0,1],[-1,0]]), np.array([0,n])),
            (np.array([[0,-1],[1,0]]), np.array([n,0])),
            (np.array([[0,-1],[-1,0]]), np.array([n,n]))]


def canonical(cells):
    n = N; pts = [tuple(map(int, p)) for p in cells]
    forms = [lambda x,y:(x,y), lambda x,y:(n-1-x,y), lambda x,y:(x,n-1-y),
             lambda x,y:(n-1-x,n-1-y), lambda x,y:(y,x), lambda x,y:(y,n-1-x),
             lambda x,y:(n-1-y,x), lambda x,y:(n-1-y,n-1-x)]
    return min(tuple(sorted(f(x,y) for x,y in pts)) for f in forms)


def stab_order(cells):
    S = frozenset(map(tuple, np.asarray(cells).tolist()))
    arr = np.asarray(cells)
    return sum(frozenset(map(tuple, (arr @ M.T + t).tolist())) == S for M, t in d4_maps())


def is_modular(cells):
    """La solucion es modular/toroidal: perm[i]=(a*i+b) mod N para algun a."""
    perm = np.zeros(N, int)
    for x, y in cells: perm[int(x)] = int(y)
    for a in range(1, N):
        if all(perm[i] == (perm[0] + a*i) % N for i in range(N)):
            return True
    return False


# ============================== estudios ====================================
def study_census(rng):
    print(f"\n[A/C/D] Censo de cuencas: {TRIALS} universos aleatorios (N={N})")
    harvest = Counter(); disp_all = []; t0 = time.time(); done = 0
    while done < TRIALS:
        bt = min(CHUNK, TRIALS-done)
        pos, vel, k, mass = make_random(bt, rng)
        ss, final, disp, _ = run_batch(pos, vel, k, mass, track_path=False)
        for i in range(bt):
            disp_all.append(disp[i].mean())
            if ss[i] >= 0:
                harvest[tuple(sorted(map(tuple, final[i].tolist())))] += 1
        done += bt
        print(f"    {done}/{TRIALS}  distintas={len(harvest)}  ({time.time()-t0:.0f}s)",
              flush=True)
    return harvest, np.array(disp_all)


def study_anchors(rng, n_anchor=4, samples=1200):
    """B1: misma posicion, energia variable.  B2: misma energia, posicion variable."""
    print(f"\n[B] Anclas: {n_anchor} anclas x {samples} muestras cada una")
    def distinct_from(fixed_pos=None, fixed_vel=None):
        counts = []
        for _ in range(n_anchor):
            pos, vel, k, mass = make_random(samples, rng)
            if fixed_pos:                       # misma posicion para todas
                pos[:] = rng.uniform(0, N-1.0, (1, N, 2))
                k[:] = np.clip(B*(1+K_SPREAD*rng.uniform(-1,1,(1,N))),0.05*B,None)
                mass[:] = 1-MASS_COUPLE+MASS_COUPLE*k/k.mean(1,keepdims=True)
            if fixed_vel:                       # misma energia (vel) para todas
                vel[:] = rng.normal(0, V0, (1, N, 2))
                k[:] = np.clip(B*(1+K_SPREAD*rng.uniform(-1,1,(1,N))),0.05*B,None)
                mass[:] = 1-MASS_COUPLE+MASS_COUPLE*k/k.mean(1,keepdims=True)
            ss, final, _, _ = run_batch(pos, vel, k, mass)
            got = {tuple(sorted(map(tuple, final[i].tolist())))
                   for i in range(samples) if ss[i] >= 0}
            solved = int((ss >= 0).sum())
            counts.append((len(got), solved))
        return counts
    b1 = distinct_from(fixed_pos=True)          # misma posicion, energia libre
    b2 = distinct_from(fixed_vel=True)          # misma energia, posicion libre
    return b1, b2


def report(harvest, disp, b1, b2):
    known = {5:10,6:4,7:40,8:92,9:352,10:724,11:2680,13:73712}
    total = known.get(N)
    print("\n" + "="*60)
    print(f"RESULTADOS  N={N}")
    print("="*60)
    n_sol = sum(harvest.values())
    print(f"cosechas totales: {n_sol}  ({100*n_sol/TRIALS:.2f}%)")
    print(f"soluciones distintas: {len(harvest)}" + (f" / {total}" if total else ""))

    # --- A: distribucion de tamanos de cuenca ---
    sizes = np.array(sorted(harvest.values(), reverse=True))
    print(f"\n[A] tamano de cuenca por solucion (cuantas iniciales -> misma final):")
    print(f"    max={sizes.max()}  mediana={int(np.median(sizes))}  min={sizes.min()}  "
          f"media={sizes.mean():.1f}")
    print(f"    razon max/min = {sizes.max()/max(sizes.min(),1):.1f}x  "
          f"(desigualdad de las cuencas)")

    # --- C: desplazamiento ---
    print(f"\n[C] desplazamiento medio por reina: {disp.mean():.2f} casillas "
          f"(N/2 = {N/2:.1f})  ->  {disp.mean()/(N/2):.2f}x de N/2")

    # --- D: estructura ---
    rows = []
    for sol, cnt in harvest.items():
        cells = np.array(sol)
        rows.append((cnt, is_modular(cells), stab_order(cells), 8 // stab_order(cells)))
    rows = np.array(rows)                       # cnt, modular, stab, orbit
    mod = rows[rows[:, 1] == 1, 0]; non = rows[rows[:, 0] >= 0][rows[:, 1] == 0, 0]
    print(f"\n[D] estructura y tamano de cuenca:")
    if len(mod) and len(non):
        print(f"    soluciones modulares/toroidales: {len(mod)}  "
              f"cuenca media = {mod.mean():.1f}")
        print(f"    soluciones NO modulares:         {len(non)}  "
              f"cuenca media = {non.mean():.1f}")
        print(f"    -> las modulares tienen cuencas {mod.mean()/max(non.mean(),1e-9):.2f}x "
              f"las de las no modulares")
    # enriquecimiento: fraccion de cosechas modulares vs tasa base de soluciones
    mod_total = {5:10, 7:28, 11:88, 13:4260}   # nº de soluciones modulares por N
    if N in mod_total and total:
        base = mod_total[N] / total
        harv_mod = int(mod.sum())
        harv_frac = harv_mod / max(n_sol, 1)
        print(f"    tasa base de soluciones modulares : {100*base:.1f}%  "
              f"({mod_total[N]}/{total})")
        print(f"    fraccion de COSECHAS modulares    : {100*harv_frac:.1f}%  "
              f"({harv_mod}/{n_sol})")
        print(f"    -> ENRIQUECIMIENTO modular = {harv_frac/max(base,1e-9):.2f}x "
              f"(el sistema prefiere la estructura si >1)")
    # correlacion cuenca vs orden de simetria
    if rows[:, 3].std() > 0 and rows[:, 0].std() > 0:
        r = np.corrcoef(rows[:, 0], rows[:, 3])[0, 1]
        print(f"    corr(tamano de cuenca, tamano de orbita D4) = {r:+.3f}")
        for orb in sorted(set(rows[:, 3])):
            m = rows[:, 3] == orb
            print(f"      orbita {orb} (simetria {8//orb}x): {m.sum():3d} sols, "
                  f"cuenca media {rows[m,0].mean():.1f}")

    # --- B ---
    b1 = np.array(b1); b2 = np.array(b2)
    if N <= 9:
        print(f"\n[B] cuantas soluciones DISTINTAS alcanza un ancla fija:")
        print(f"    misma POSICION, energia variable : {b1[:,0].mean():.1f} distintas "
              f"(de {b1[:,1].mean():.0f} resueltas de media)")
        print(f"    misma ENERGIA, posicion variable : {b2[:,0].mean():.1f} distintas "
              f"(de {b2[:,1].mean():.0f} resueltas de media)")
    return sizes, rows


def figures(harvest, disp, sizes, rows, b1, b2):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("(sin matplotlib)", e); return
    plt.rcParams.update({"figure.facecolor":"#0d1117","axes.facecolor":"#0d1117",
        "text.color":"#e6edf6","axes.labelcolor":"#8ea0b6","xtick.color":"#8ea0b6",
        "ytick.color":"#8ea0b6","axes.edgecolor":"#30363d","axes.titlecolor":"#4fe0cf",
        "font.size":9})
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Newton Queens -- cuencas de atraccion N={N} ({TRIALS} universos)",
                 color="#4fe0cf", fontsize=14)

    ax[0,0].bar(range(1, len(sizes)+1), sizes, color="#4fe0cf")
    ax[0,0].set_title("tamano de cuenca por solucion (ordenado)")
    ax[0,0].set_xlabel("solucion (rango)"); ax[0,0].set_ylabel("# iniciales -> esa final")

    ax[0,1].hist(sizes, bins=25, color="#7b6cff")
    ax[0,1].set_title("distribucion de tamanos de cuenca")
    ax[0,1].set_xlabel("tamano de cuenca"); ax[0,1].set_ylabel("# soluciones")

    ax[0,2].hist(disp, bins=40, color="#ffb454")
    ax[0,2].axvline(disp.mean(), color="#ff556e", ls="--", label=f"media {disp.mean():.2f}")
    ax[0,2].axvline(N/2, color="#77e6a0", ls=":", label=f"N/2 = {N/2:.1f}")
    ax[0,2].set_title("desplazamiento medio por universo"); ax[0,2].legend(
        facecolor="#131c28", labelcolor="#e6edf6")
    ax[0,2].set_xlabel("casillas")

    # modular vs no modular
    mod = rows[rows[:,1]==1,0]; non = rows[rows[:,1]==0,0]
    ax[1,0].boxplot([non, mod] if len(mod) and len(non) else [sizes],
                    tick_labels=["no modular","modular"] if len(mod) and len(non) else ["todas"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#1c6f68"), medianprops=dict(color="#ffb454"))
    ax[1,0].set_title("cuenca: modular vs no modular"); ax[1,0].set_ylabel("tamano de cuenca")

    # cuenca vs orbita
    for orb in sorted(set(rows[:,3])):
        m = rows[:,3]==orb
        ax[1,1].scatter(rows[m,3]+np.random.uniform(-.15,.15,m.sum()), rows[m,0],
                        alpha=.5, s=25, color="#ff9de2")
    ax[1,1].set_title("cuenca vs tamano de orbita D4")
    ax[1,1].set_xlabel("tamano de orbita (8/simetria)"); ax[1,1].set_ylabel("tamano de cuenca")

    b1 = np.array(b1); b2 = np.array(b2)
    ax[1,2].bar(["misma pos\n(energia libre)","misma energia\n(pos libre)"],
                [b1[:,0].mean(), b2[:,0].mean()],
                yerr=[b1[:,0].std(), b2[:,0].std()], color=["#4fe0cf","#ffb454"], capsize=6)
    ax[1,2].set_title("soluciones distintas alcanzadas por ancla fija")
    ax[1,2].set_ylabel("# soluciones distintas")

    fig.tight_layout(rect=[0,0,1,0.97])
    out = f"basin_N{N}.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigura -> {out}")


if __name__ == "__main__":
    print(f"# estudio de cuencas  N={N}  TRIALS={TRIALS}  MAX_STEPS={MAX_STEPS}  V0={V0}")
    rng = np.random.default_rng(SEED)
    harvest, disp = study_census(rng)
    # el estudio de anclas solo tiene sentido con pocas soluciones (N pequeno)
    if N <= 9:
        b1, b2 = study_anchors(rng)
    else:
        print("\n[B] omitido (espacio de soluciones demasiado grande para N>=10)")
        b1, b2 = [(0, 0)], [(0, 0)]
    sizes, rows = report(harvest, disp, b1, b2)
    figures(harvest, disp, sizes, rows, b1, b2)
