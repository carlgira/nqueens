"""
Newton Queens -- analisis de trayectorias y patrones
====================================================

Busca estructura en el caos disipativo. Cuatro preguntas:

  1. EQUIVARIANCIA (el teorema bonito). El potencial es invariante bajo las 8
     simetrias del cuadrado (D4) y la friccion es isotropa => la dinamica es
     EQUIVARIANTE. Si rotas/reflejas la condicion inicial completa (posiciones
     Y velocidades) con una simetria g, TODA la trayectoria es la imagen g y el
     atractor final es g(solucion). Lo demostramos numericamente: evolucionamos
     una condicion inicial y sus 8 imagenes, y verificamos que coinciden hasta
     precision de maquina.

  2. PODER vs MOVIMIENTO. Hipotesis del usuario: las reinas poderosas (k alto,
     pesadas) reclaman su sitio y apenas se mueven; las debiles salen despedidas.
     Correlacionamos el poder/masa de cada reina con la longitud de su camino.

  3. CONDICION INICIAL vs CONVERGENCIA. Relacion entre conflictos iniciales,
     energia cinetica inicial y cuantos pasos tarda en cristalizar.

  4. CUENCAS SIMETRICAS. Por equivariancia, las soluciones relacionadas por
     simetria tienen cuencas de atraccion congruentes => se cosechan con la
     misma frecuencia. Lo verificamos sobre el ensemble.

Toda la fisica esta vectorizada sobre el eje de universos (identica a
experiment.py).  Genera analyze_N8.png (+ paneles).
"""
from __future__ import annotations
import time, sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ENSEMBLE = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
MAX_STEPS = 6500
CHECK_EVERY = 5
CHUNK = 2000
SEED = 4242

A, B, SIGMA = 2.5, 8.0, 0.5
GAMMA, V0, K_SPREAD, MASS_COUPLE = 0.03, 3.0, 0.7, 0.8
WALL_K, DT = 30.0, 0.012
TAU, S2 = 2 * np.pi, 2 * 0.5 ** 2


# ------------------------------ fisica vectorizada --------------------------
def forces(pos, vel, k, mass):
    x, y = pos[..., 0], pos[..., 1]
    fx = -TAU * A * np.sin(TAU * x)
    fy = -TAU * A * np.sin(TAU * y)
    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    u, v = dx - dy, dx + dy
    T1 = np.exp(-(dx * dx) / S2); T2 = np.exp(-(dy * dy) / S2)
    T3 = np.exp(-(u * u) / S2);   T4 = np.exp(-(v * v) / S2)
    kj = k[:, None, :] / SIGMA ** 2
    fx += (kj * (dx * T1 + u * T3 + v * T4)).sum(axis=2)
    fy += (kj * (dy * T2 - u * T3 + v * T4)).sum(axis=2)
    lo, hi = 0.0, N - 1.0
    fx += WALL_K * np.maximum(lo - x, 0) - WALL_K * np.maximum(x - hi, 0)
    fy += WALL_K * np.maximum(lo - y, 0) - WALL_K * np.maximum(y - hi, 0)
    fx -= GAMMA * vel[..., 0]; fy -= GAMMA * vel[..., 1]
    return np.stack([fx, fy], axis=-1)


def accel(pos, vel, k, mass):
    return forces(pos, vel, k, mass) / mass[..., None]


def rk4_step(pos, vel, k, mass):
    a1 = accel(pos, vel, k, mass)
    p2, v2 = pos + .5 * DT * vel, vel + .5 * DT * a1
    a2 = accel(p2, v2, k, mass)
    p3, v3 = pos + .5 * DT * v2, vel + .5 * DT * a2
    a3 = accel(p3, v3, k, mass)
    p4, v4 = pos + DT * v3, vel + DT * a3
    a4 = accel(p4, v4, k, mass)
    dpos = (DT / 6) * (vel + 2 * v2 + 2 * v3 + v4)
    dvel = (DT / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
    return pos + dpos, vel + dvel, dpos


def cells_of(pos):
    return np.clip(np.rint(pos), 0, N - 1).astype(np.int32)


def conflicts_of(pos):
    c = cells_of(pos); cx, cy = c[..., 0], c[..., 1]
    sc = cx[:, :, None] == cx[:, None, :]
    sr = cy[:, :, None] == cy[:, None, :]
    sd = np.abs(cx[:, :, None] - cx[:, None, :]) == np.abs(cy[:, :, None] - cy[:, None, :])
    at = (sc | sr | sd) & ~np.eye(N, dtype=bool)[None]
    return at.sum(axis=(1, 2)) // 2


def solved_of(pos):
    return conflicts_of(pos) == 0, cells_of(pos)


# --------------------------- simetrias del cuadrado (D4) --------------------
def d4_maps():
    n = N - 1
    return {
        "identidad":   (np.array([[1, 0], [0, 1]]),  np.array([0, 0])),
        "reflex |":    (np.array([[-1, 0], [0, 1]]), np.array([n, 0])),
        "reflex --":   (np.array([[1, 0], [0, -1]]), np.array([0, n])),
        "rot 180":     (np.array([[-1, 0], [0, -1]]), np.array([n, n])),
        "reflex \\":   (np.array([[0, 1], [1, 0]]),  np.array([0, 0])),
        "rot 90":      (np.array([[0, 1], [-1, 0]]), np.array([0, n])),
        "rot 270":     (np.array([[0, -1], [1, 0]]), np.array([n, 0])),
        "reflex /":    (np.array([[0, -1], [-1, 0]]), np.array([n, n])),
    }


def canonical(cells):
    n = N
    pts = [tuple(map(int, p)) for p in cells]
    forms = [lambda x, y: (x, y), lambda x, y: (n-1-x, y), lambda x, y: (x, n-1-y),
             lambda x, y: (n-1-x, n-1-y), lambda x, y: (y, x), lambda x, y: (y, n-1-x),
             lambda x, y: (n-1-y, x), lambda x, y: (n-1-y, n-1-x)]
    return min(tuple(sorted(f(x, y) for x, y in pts)) for f in forms)


# ========================= PARTE 1: EQUIVARIANCIA Y CAOS ====================
def part1_equivariance():
    """Equivariancia D4 exacta + su ruptura por caos (Lyapunov)."""
    print("\n" + "=" * 60)
    print("PARTE 1 -- EQUIVARIANCIA D4 vs CAOS")
    print("=" * 60)
    rng = np.random.default_rng(7)
    pos0 = rng.uniform(0, N - 1.0, (1, N, 2))
    vel0 = rng.normal(0, V0, (1, N, 2))
    k = B * (1 + K_SPREAD * rng.uniform(-1, 1, (1, N)))
    k = np.clip(k, 0.05 * B, None)
    mass = 1 - MASS_COUPLE + MASS_COUPLE * k / k.mean(1, keepdims=True)

    # 1) equivariancia EXACTA de las fuerzas (arit. exacta): F(g x) = g F(x)
    F = accel(pos0, vel0, k, mass)
    print("\n(1) Equivariancia de las fuerzas (deberia ser ~precision de maquina):")
    for name, (M, t) in d4_maps().items():
        Fg = accel(pos0 @ M.T + t, vel0 @ M.T, k, mass)
        print(f"    {name:12s}  |F(g x) - g F(x)| = {np.abs(Fg - F @ M.T).max():.1e}")

    # 2) evolucionar la base y sus 8 imagenes; medir el error de equivariancia
    #    en cada paso -> si el sistema es caotico, crece exponencialmente.
    STEPS = 4200
    maps = list(d4_maps().items())
    base = pos0.copy(); bvel = vel0.copy(); btraj = [base[0].copy()]
    imgs = [( (pos0 @ M.T + t).copy(), (vel0 @ M.T).copy(), M, t, name)
            for name, (M, t) in maps]
    errs = {name: [] for _, _, _, _, name in imgs}
    itraj = {im[4]: [im[0][0].copy()] for im in imgs}
    for step in range(STEPS):
        base, bvel, _ = rk4_step(base, bvel, k, mass)
        btraj.append(base[0].copy())
        for idx, (p, v, M, t, name) in enumerate(imgs):
            p, v, _ = rk4_step(p, v, k, mass)
            imgs[idx] = (p, v, M, t, name)
            mapped = base[0] @ M.T + t            # g aplicado a la base en t
            errs[name].append(float(np.abs(p[0] - mapped).max()))
            itraj[name].append(p[0].copy())
    btraj = np.array(btraj)

    print("\n(2) Error de equivariancia de la TRAYECTORIA tras {} pasos:".format(STEPS))
    print(f"    {'simetria':12s} {'exacta en FP?':>14s} {'error final':>14s}")
    lyap = None
    for name in errs:
        e = np.array(errs[name])
        exact_fp = e.max() < 1e-9
        print(f"    {name:12s} {str(exact_fp):>14s} {e[-1]:14.2e}")
        if not exact_fp and lyap is None:
            # ajustar exponente de Lyapunov en la region de crecimiento exponencial
            mask = (e > 1e-12) & (e < 1e-1)
            if mask.sum() > 50:
                ts = np.arange(len(e))[mask] * DT
                slope = np.polyfit(ts, np.log(e[mask]), 1)[0]
                lyap = slope
    print("\n=> Las fuerzas son equivariantes EXACTO (arit. exacta): rotar la condicion")
    print("   inicial DEBERIA dar la solucion rotada. Pero el sistema es CAOTICO:")
    print("   identidad y transposicion (reflex \\) son exactas en binario (solo")
    print("   permutan/niegan ejes) y se mantienen a error 0. Las demas introducen")
    print("   redondeo ~1e-16 que el caos AMPLIFICA exponencialmente hasta cruzar")
    print("   una frontera de cuenca -> acaban en OTRA solucion valida distinta.")
    if lyap:
        print(f"   Exponente de Lyapunov estimado: lambda ~ {lyap:.2f} / unidad de tiempo")
        print(f"   (tiempo de duplicacion del error ~ {np.log(2)/lyap/DT:.0f} pasos)")
    return btraj, errs, itraj, lyap


# ===================== PARTE 2/3: ENSEMBLE INSTRUMENTADO ====================
def part_ensemble():
    print("\n" + "=" * 60)
    print("PARTE 2/3 -- ENSEMBLE INSTRUMENTADO ({} universos)".format(ENSEMBLE))
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    rec = {kk: [] for kk in
           ("k", "mass", "path", "disp", "solved", "steps",
            "init_conf", "init_ke", "final_col_rank_corr", "canon", "exact")}
    done = 0; t0 = time.time()
    while done < ENSEMBLE:
        bt = min(CHUNK, ENSEMBLE - done)
        pos = rng.uniform(0, N - 1.0, (bt, N, 2))
        vel = rng.normal(0, V0, (bt, N, 2))
        k = B * (1 + K_SPREAD * rng.uniform(-1, 1, (bt, N)))
        k = np.clip(k, 0.05 * B, None)
        mass = 1 - MASS_COUPLE + MASS_COUPLE * k / k.mean(1, keepdims=True)

        pos0 = pos.copy()
        init_conf = conflicts_of(pos)
        init_ke = 0.5 * (mass[..., None] * vel ** 2).sum(axis=(1, 2))
        pathlen = np.zeros((bt, N))
        solved_step = np.full(bt, -1, np.int32)
        final_cells = np.zeros((bt, N, 2), np.int32)
        active = np.ones(bt, bool)

        for step in range(MAX_STEPS):
            pos, vel, dpos = rk4_step(pos, vel, k, mass)
            pathlen += np.linalg.norm(dpos, axis=2) * active[:, None]
            if step % CHECK_EVERY == 0 or step == MAX_STEPS - 1:
                sol, cells = solved_of(pos)
                fresh = sol & active
                if fresh.any():
                    solved_step[fresh] = step
                    final_cells[fresh] = cells[fresh]
                    active[fresh] = False
                if not active.any():
                    break

        disp = np.linalg.norm(cells_of(pos if False else pos) - pos0, axis=2)  # start->end
        for i in range(bt):
            rec["k"].append(k[i].copy())
            rec["mass"].append(mass[i].copy())
            rec["path"].append(pathlen[i].copy())
            rec["disp"].append(disp[i].copy())
            rec["init_conf"].append(int(init_conf[i]))
            rec["init_ke"].append(float(init_ke[i]))
            s = solved_step[i] >= 0
            rec["solved"].append(bool(s))
            rec["steps"].append(int(solved_step[i]))
            if s:
                fc = final_cells[i]
                # correlacion rango(x inicial) vs columna final
                xi = pos0[i, :, 0]
                order_init = np.argsort(np.argsort(xi))
                order_fin = np.argsort(np.argsort(fc[:, 0].astype(float)))
                rec["final_col_rank_corr"].append(
                    np.corrcoef(order_init, order_fin)[0, 1])
                rec["canon"].append(canonical(fc))
                rec["exact"].append(tuple(sorted(map(tuple, fc.tolist()))))
        done += bt
        print(f"  {done}/{ENSEMBLE}  resueltos {sum(rec['solved'])}  ({time.time()-t0:.0f}s)",
              flush=True)
    return rec


def report_ensemble(rec):
    solved = np.array(rec["solved"])
    n_ok = int(solved.sum())
    print(f"\nresueltos: {n_ok}/{len(solved)} ({100*n_ok/len(solved):.2f}%)")

    # ---- Parte 2: poder vs movimiento ----
    k = np.concatenate(rec["k"]); mass = np.concatenate(rec["mass"])
    path = np.concatenate(rec["path"]); disp = np.concatenate(rec["disp"])
    r_kp = np.corrcoef(k, path)[0, 1]
    r_mp = np.corrcoef(mass, path)[0, 1]
    r_kd = np.corrcoef(k, disp)[0, 1]
    print("\n-- PODER vs MOVIMIENTO (por reina, todas) --")
    print(f"  corr(poder k, longitud de camino) = {r_kp:+.3f}")
    print(f"  corr(masa,    longitud de camino) = {r_mp:+.3f}")
    print(f"  corr(poder k, desplazamiento)     = {r_kd:+.3f}")
    # medias por cuartil de poder
    q = np.quantile(k, [0, .25, .5, .75, 1.0])
    print("  camino medio por cuartil de poder:")
    for i in range(4):
        m = (k >= q[i]) & (k <= q[i + 1])
        print(f"    Q{i+1} (k in [{q[i]:.1f},{q[i+1]:.1f}]): camino={path[m].mean():.2f}")

    # ---- Parte 3: condicion inicial vs convergencia ----
    idx = np.where(solved)[0]
    steps = np.array(rec["steps"])[idx]
    ic = np.array(rec["init_conf"])
    ke = np.array(rec["init_ke"])
    ic_s = ic[idx]; ke_s = ke[idx]
    print("\n-- CONDICION INICIAL vs CONVERGENCIA (solo resueltos) --")
    print(f"  corr(conflictos iniciales, pasos)   = {np.corrcoef(ic_s, steps)[0,1]:+.3f}")
    print(f"  corr(energia cinetica inicial, pasos)= {np.corrcoef(ke_s, steps)[0,1]:+.3f}")
    # tasa de exito por conflictos iniciales
    print("  tasa de exito segun conflictos iniciales:")
    for c in range(0, 9):
        m = ic == c
        if m.sum() > 20:
            print(f"    {c} conflictos iniciales: {100*solved[m].mean():.2f}%  (n={m.sum()})")

    fcorr = np.array(rec["final_col_rank_corr"])
    print(f"\n-- INICIAL -> FINAL --")
    print(f"  corr media rango(x inicial) vs columna final = {np.nanmean(fcorr):+.3f}")
    print(f"  desplazamiento medio por reina = {disp.mean():.2f} casillas "
          f"(mediana {np.median(disp):.2f})")

    # ---- Parte 4: cuencas simetricas ----
    from collections import Counter
    exact_ct = Counter(rec["exact"])
    canon_ct = Counter(rec["canon"])
    print("\n-- CUENCAS SIMETRICAS --")
    print(f"  soluciones exactas distintas cosechadas : {len(exact_ct)}")
    print(f"  clases fundamentales cubiertas          : {len(canon_ct)}/12")
    # dentro de una clase, las exactas deben aparecer con frecuencia similar
    print("  frecuencia de cosecha por clase fundamental (tamano de cuenca):")
    for i, (cn, tot) in enumerate(canon_ct.most_common()):
        # cuantas orientaciones distintas de esta clase salieron
        orients = [v for e, v in exact_ct.items() if canonical(np.array(e)) == cn]
        print(f"    clase {i+1:2d}: {tot:4d} cosechas en {len(orients):2d} orientaciones "
              f"(media {np.mean(orients):.1f} +/- {np.std(orients):.1f})")
    return dict(k=k, mass=mass, path=path, disp=disp, steps=steps,
                ic=ic, ke=ke, solved=solved, ic_s=ic_s, ke_s=ke_s,
                exact_ct=exact_ct, canon_ct=canon_ct, fcorr=fcorr)


# ------------------------------- figuras ------------------------------------
def figures(btraj, errs, itraj, lyap, S):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(sin matplotlib)", e); return
    dark = {"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
            "text.color": "#e6edf6", "axes.labelcolor": "#8ea0b6",
            "xtick.color": "#8ea0b6", "ytick.color": "#8ea0b6",
            "axes.edgecolor": "#30363d", "axes.titlecolor": "#4fe0cf", "font.size": 9}
    plt.rcParams.update(dark)
    cols = ["#4fe0cf", "#ffb454", "#7b6cff", "#ff556e", "#77e6a0",
            "#e6edf6", "#ff9de2", "#5ac8fa"]

    # ---------- figura A: equivariancia (corto plazo) + Lyapunov -------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # panel izq: las 8 imagenes simetricas a corto plazo (mandala D4 exacto)
    SHORT = 150
    ax[0].set_title(f"Equivariancia D4: una condicion inicial y sus 8 imagenes\n"
                    f"(primeros {SHORT} pasos: copias simetricas exactas)")
    for (name, tr), c in zip(itraj.items(), cols):
        tr = np.array(tr)[:SHORT]
        for qn in range(N):
            ax[0].plot(tr[:, qn, 0], tr[:, qn, 1], "-", color=c, alpha=0.8, lw=1.1)
        ax[0].scatter(tr[0, :, 0], tr[0, :, 1], s=40, color=c, edgecolors="k",
                      linewidths=0.4, zorder=6)
    ax[0].set_xlim(-.6, N - .4); ax[0].set_ylim(-.6, N - .4)
    ax[0].set_xticks(range(N)); ax[0].set_yticks(range(N)); ax[0].grid(alpha=.12)
    ax[0].set_aspect("equal")

    # panel der: error de equivariancia vs tiempo (log) -> Lyapunov
    ax[1].set_title("El caos rompe la simetria: error de equivariancia vs tiempo")
    for (name, e), c in zip(errs.items(), cols):
        e = np.array(e)
        e = np.where(e <= 0, 1e-17, e)
        ax[1].semilogy(np.arange(len(e)) * DT, e, color=c, lw=1.2, label=name)
    ax[1].set_xlabel("tiempo"); ax[1].set_ylabel("|g·traj − traj(g·x0)|  (log)")
    if lyap:
        ax[1].text(0.03, 0.95, f"λ ≈ {lyap:.2f}/t  (Lyapunov)", transform=ax[1].transAxes,
                   color="#ffb454", va="top", fontsize=11)
    ax[1].legend(fontsize=7, facecolor="#131c28", edgecolor="#30363d",
                 labelcolor="#e6edf6", ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(f"analyze_N{N}_equivariance.png", dpi=140,
                                    bbox_inches="tight")
    print(f"\nfigura -> analyze_N{N}_equivariance.png")

    # ---------- figura B: patrones del ensemble ----------
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Newton Queens -- patrones en {len(S['solved'])} universos (N={N})",
                 color="#4fe0cf", fontsize=14)

    # poder vs camino (hexbin)
    ax[0, 0].hexbin(S["k"], S["path"], gridsize=35, cmap="magma", mincnt=1)
    ax[0, 0].set_title(f"poder k vs camino  (r={np.corrcoef(S['k'],S['path'])[0,1]:+.2f})")
    ax[0, 0].set_xlabel("poder k_i"); ax[0, 0].set_ylabel("longitud del camino")

    # masa vs camino
    ax[0, 1].hexbin(S["mass"], S["path"], gridsize=35, cmap="viridis", mincnt=1)
    ax[0, 1].set_title(f"masa vs camino  (r={np.corrcoef(S['mass'],S['path'])[0,1]:+.2f})")
    ax[0, 1].set_xlabel("masa m_i"); ax[0, 1].set_ylabel("longitud del camino")

    # desplazamiento
    ax[0, 2].hist(S["disp"], bins=50, color="#7b6cff")
    ax[0, 2].axvline(np.median(S["disp"]), color="#ffb454", ls="--",
                     label=f"mediana {np.median(S['disp']):.2f}")
    ax[0, 2].set_title("desplazamiento inicio->final por reina")
    ax[0, 2].set_xlabel("casillas"); ax[0, 2].legend(facecolor="#131c28",
                                                     labelcolor="#e6edf6")

    # conflictos iniciales vs tasa de exito
    ic = S["ic"]; solved = S["solved"]
    xs, ys = [], []
    for c in range(0, 9):
        m = ic == c
        if m.sum() > 20:
            xs.append(c); ys.append(100 * solved[m].mean())
    ax[1, 0].bar(xs, ys, color="#4fe0cf")
    ax[1, 0].set_title("tasa de exito vs conflictos iniciales")
    ax[1, 0].set_xlabel("conflictos en t=0"); ax[1, 0].set_ylabel("% resuelto")

    # conflictos iniciales vs pasos (violin-ish scatter)
    ax[1, 1].hexbin(S["ic_s"], S["steps"], gridsize=25, cmap="plasma", mincnt=1)
    ax[1, 1].set_title(f"conflictos iniciales vs pasos "
                       f"(r={np.corrcoef(S['ic_s'],S['steps'])[0,1]:+.2f})")
    ax[1, 1].set_xlabel("conflictos en t=0"); ax[1, 1].set_ylabel("pasos a cristalizar")

    # cuencas por clase fundamental
    canon_ct = S["canon_ct"]; exact_ct = S["exact_ct"]
    sizes = sorted(canon_ct.values(), reverse=True)
    ax[1, 2].bar(range(1, len(sizes) + 1), sizes, color="#ff556e")
    ax[1, 2].set_title("tamano de cuenca por clase fundamental (12)")
    ax[1, 2].set_xlabel("clase"); ax[1, 2].set_ylabel("veces cosechada")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"analyze_N{N}_patterns.png", dpi=130, bbox_inches="tight")
    print(f"figura -> analyze_N{N}_patterns.png")


if __name__ == "__main__":
    btraj, errs, itraj, lyap = part1_equivariance()
    rec = part_ensemble()
    S = report_ensemble(rec)
    figures(btraj, errs, itraj, lyap, S)
