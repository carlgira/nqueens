"""Look for structure in the waves of a stored featured run.  Never simulates.

    python -m wave_queens.structure winner_n7

Two questions:
  waves      -- what do the N signals look like, and is there anything in their
                spectrum or in the Fourier modes of their superposition?
  itinerary  -- how does the orbit move between solutions?  Are modular ones
                stickier than non-modular ones?  Are the transitions local?
"""

import collections
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .featured import load

R = "wave_queens/results"


def order_params(q, n):
    """|S_m(t)| for m=1..N-1: the Fourier modes of the column superposition.

    S_m = sum_k exp(2 pi i m x_k / N).  All of them vanish exactly when the
    queens sit on N distinct columns -- this is the 'sum of the waves is flat'
    statement in its original form, one number per mode.
    """
    m = np.arange(1, n)
    ph = np.exp(2j * np.pi * np.einsum("m,tk->tmk", m, q) / n)
    return np.abs(ph.sum(axis=2)) / n


def fig_waves(meta, path):
    tr, n = meta["traces"], meta["config"]["n"]
    t, V, q, prof, grid = (tr["t"], tr["V"], tr["q"], tr["profile"], tr["grid"])
    vis = meta["visits"]
    S = order_params(q, n)

    fig, ax = plt.subplots(4, 1, figsize=(13, 12), sharex=False)

    ax[0].imshow(prof.T, aspect="auto", origin="lower", cmap="magma",
                 extent=[t[0], t[-1], grid[0], grid[-1]])
    ax[0].set_ylabel("column")
    ax[0].set_title("superposition profile of the cold replica: uniform stripes = "
                    "flat sum = a solution on the board", fontsize=10)

    ax[1].plot(t, V, lw=.5, color="0.3")
    for v in vis:
        if v["replica"] == 0:
            ax[1].axvspan(v["t_start"], v["t_end"], color=
                          "tab:blue" if v["modular"] else "tab:orange", alpha=.45)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("V  (flatness witness)")
    ax[1].set_title("blue = modular solution on the board, orange = non-modular",
                    fontsize=10)

    for k in range(n):
        ax[2].plot(t, q[:, k], lw=.6, label=f"q{k}")
    ax[2].set_ylabel("queen position")
    ax[2].set_xlabel("time")
    ax[2].set_title("the N waves themselves", fontsize=10)

    for i in range(S.shape[1]):
        ax[3].plot(t, S[:, i], lw=.5, label=f"|S_{i+1}|")
    ax[3].set_ylabel("|S_m| / N")
    ax[3].set_xlabel("time")
    ax[3].legend(fontsize=7, ncol=n)
    ax[3].set_title("Fourier modes of the superposition -- all zero together "
                    "means the sum is flat", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return S


def fig_itinerary(meta, S, path):
    n = meta["config"]["n"]
    tr = meta["traces"]
    vis = meta["visits"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # (1) itinerary: which solution, when, from which replica
    order, seen = {}, []
    for v in vis:
        key = tuple(v["placement"])
        if key not in order:
            order[key] = len(order)
            seen.append(v)
    for v in vis:
        y = order[tuple(v["placement"])]
        ax[0][0].plot([v["t_start"], max(v["t_end"], v["t_start"] + .5)], [y, y],
                      lw=3, color="tab:blue" if v["modular"] else "tab:orange",
                      solid_capstyle="butt")
    ax[0][0].set_xlabel("time")
    ax[0][0].set_ylabel("solution (order of discovery)")
    ax[0][0].set_title(f"itinerary: {len(vis)} visits over {len(order)} solutions",
                       fontsize=10)

    # (2) dwell time, modular vs not -- is one kind stickier?
    dm = [v["t_end"] - v["t_start"] for v in vis if v["modular"]]
    dn = [v["t_end"] - v["t_start"] for v in vis if not v["modular"]]
    bins = np.logspace(-1.4, np.log10(max(dm + dn + [1]) + 1), 22)
    ax[0][1].hist([dm, dn], bins=bins, color=["tab:blue", "tab:orange"],
                  label=[f"modular (n={len(dm)}, med={np.median(dm or [0]):.2f})",
                         f"non-modular (n={len(dn)}, med={np.median(dn or [0]):.2f})"])
    ax[0][1].set_xscale("log")
    ax[0][1].set_xlabel("dwell time")
    ax[0][1].set_ylabel("visits")
    ax[0][1].legend(fontsize=8)
    ax[0][1].set_title("how long it rests, by kind of solution", fontsize=10)

    # (3) how far does it jump?  queens moved between consecutive visits
    per_rep = collections.defaultdict(list)
    for v in vis:
        per_rep[v["replica"]].append(v)
    hops = []
    for vs in per_rep.values():
        for a, b in zip(vs, vs[1:]):
            if a["placement"] != b["placement"]:
                hops.append(sum(x != y for x, y in
                                zip(a["placement"], b["placement"])))
    if hops:
        ax[1][0].hist(hops, bins=np.arange(1.5, n + 1.5), color="tab:purple")
    ax[1][0].set_xlabel("queens that changed column")
    ax[1][0].set_ylabel("transitions")
    ax[1][0].set_title(f"size of a hop between consecutive solutions "
                       f"(n={len(hops)})", fontsize=10)

    # (4) which Fourier mode is the last to flatten?
    flat = S.min(axis=0)
    resid = S[S.sum(axis=1) > np.percentile(S.sum(axis=1), 60)].mean(axis=0)
    x = np.arange(1, n)
    ax[1][1].bar(x - .2, flat, .4, label="min over run (flattest)", color="tab:green")
    ax[1][1].bar(x + .2, resid, .4, label="mean in the noisy phase", color="0.6")
    ax[1][1].set_xlabel("Fourier mode m")
    ax[1][1].set_ylabel("|S_m| / N")
    ax[1][1].legend(fontsize=8)
    ax[1][1].set_title("which mode resists going flat", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return hops, dm, dn


def fig_zoom(meta, path, span=24.0):
    """Panel 3 of the waves figure is unreadable over t=1200.  Zoom on the
    longest plateau of the cold replica to see what the waves actually do."""
    tr, n = meta["traces"], meta["config"]["n"]
    t, V, q, prof, grid = tr["t"], tr["V"], tr["q"], tr["profile"], tr["grid"]
    cold = [v for v in meta["visits"] if v["replica"] == 0]
    best = max(cold, key=lambda v: v["t_end"] - v["t_start"])
    c = 0.5 * (best["t_start"] + best["t_end"])
    sel = (t > c - span / 2) & (t < c + span / 2)
    S = order_params(q[sel], n)

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].imshow(prof[sel].T, aspect="auto", origin="lower", cmap="magma",
                 extent=[t[sel][0], t[sel][-1], grid[0], grid[-1]])
    ax[0].set_ylabel("column")
    ax[0].set_title(f"zoom on the longest plateau of the cold replica "
                    f"({'modular' if best['modular'] else 'non-modular'}: "
                    f"{best['placement']}, dwell "
                    f"{best['t_end']-best['t_start']:.2f})", fontsize=10)
    for k in range(n):
        ax[1].plot(t[sel], q[sel][:, k], lw=1.1)
    ax[1].axvspan(best["t_start"], best["t_end"], color="tab:green", alpha=.18)
    for lev in range(n):
        ax[1].axhline(lev, color="0.85", lw=.5, zorder=0)
    ax[1].set_ylabel("queen position")
    ax[1].set_title("the N waves: green band = an exact solution is on the board",
                    fontsize=10)
    ax[2].plot(t[sel], V[sel], color="0.3", lw=1)
    ax[2].axvspan(best["t_start"], best["t_end"], color="tab:green", alpha=.18)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("V")
    ax[2].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return best


def fig_green(meta, path, window=(240.0, 300.0)):
    """The wave that carries the most solutions, with every solution in green.

    Only the cold replica's trajectory is stored, and it happens to be one of
    the richest: 11 of the run's 18 distinct solutions over 277 visits.  Top
    two panels are the whole run, bottom two zoom on the densest stretch.
    """
    tr, n = meta["traces"], meta["config"]["n"]
    t, V, q, prof, grid = tr["t"], tr["V"], tr["q"], tr["profile"], tr["grid"]
    vis = [v for v in meta["visits"] if v["replica"] == 0]
    uniq = {tuple(v["placement"]) for v in vis}

    def bands(ax, lo, hi, label=False):
        seen = {}
        for v in vis:
            if v["t_end"] < lo or v["t_start"] > hi:
                continue
            ax.axvspan(v["t_start"], max(v["t_end"], v["t_start"] + 0.06),
                       color="tab:green", alpha=.45, lw=0)
            k = tuple(v["placement"])
            if label and k not in seen:
                seen[k] = v["t_start"]
        return seen

    fig, ax = plt.subplots(4, 1, figsize=(14, 13))

    for k in range(n):
        ax[0].plot(t, q[:, k], lw=.35)
    bands(ax[0], t[0], t[-1])
    ax[0].set_ylabel("queen position")
    ax[0].set_title(f"whole run: the {n} waves, green = an exact solution is on "
                    f"the board ({len(vis)} visits over {len(uniq)} distinct "
                    f"solutions)", fontsize=10)
    ax[0].axvspan(*window, facecolor="none", edgecolor="tab:red", lw=1.5)

    ax[1].plot(t, V, lw=.4, color="0.3")
    bands(ax[1], t[0], t[-1])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("V (flatness witness)")
    ax[1].set_xlabel("time")
    ax[1].axvspan(*window, facecolor="none", edgecolor="tab:red", lw=1.5)
    ax[1].set_title("same run, the witness -- every green band is a dip",
                    fontsize=10)

    sel = (t >= window[0]) & (t <= window[1])
    ax[2].imshow(prof[sel].T, aspect="auto", origin="lower", cmap="magma",
                 extent=[window[0], window[1], grid[0], grid[-1]])
    ax[2].set_ylabel("column")
    ax[2].set_title(f"zoom {window[0]:.0f}-{window[1]:.0f}: superposition profile "
                    f"-- clean parallel stripes = flat sum", fontsize=10)

    for k in range(n):
        ax[3].plot(t[sel], q[sel][:, k], lw=1.0)
    seen = bands(ax[3], *window, label=True)
    for i, (k, t0) in enumerate(sorted(seen.items(), key=lambda kv: kv[1])):
        ax[3].annotate("".join(str(c) for c in k), (t0, n - 0.35),
                       fontsize=7, rotation=90, ha="center", va="top",
                       color="darkgreen")
    for lev in range(n):
        ax[3].axhline(lev, color="0.88", lw=.5, zorder=0)
    ax[3].set_ylabel("queen position")
    ax[3].set_xlabel("time")
    ax[3].set_title(f"same window, the waves: {len(seen)} distinct solutions "
                    f"(labelled), each reached by a full rescramble", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return vis, uniq


FAMILY_COLORS = ["tab:blue", "tab:cyan", "tab:green", "tab:olive",   # modular
                 "tab:orange", "tab:red"]                            # non-modular


def families(n=7):
    """The symmetry classes of standard N-queens, ordered modular-first.

    For N=7 no class mixes the two kinds: 4 classes are purely modular (28
    solutions) and 2 are purely non-modular (12), so a family colour also reads
    as a modular/non-modular colour.
    """
    from .core import canonical, enumerate_solutions
    std = [tuple(x) for x in enumerate_solutions(n, modular=False)]
    mod = {tuple(x) for x in enumerate_solutions(n, modular=True)}
    tally = collections.defaultdict(lambda: [0, 0])
    for x in std:
        tally[canonical(np.array(x), n, False)][0 if x in mod else 1] += 1
    order = sorted(tally, key=lambda k: (-tally[k][0], -sum(tally[k])))
    return {k: dict(idx=i, label=f"C{i+1}", size=sum(tally[k]),
                    modular=tally[k][0] > 0, color=FAMILY_COLORS[i])
            for i, k in enumerate(order)}


def fig_by_family(meta, path, window=(240.0, 300.0)):
    """Same picture as fig_green, but one colour per solution family."""
    tr, n = meta["traces"], meta["config"]["n"]
    t, V, q, prof, grid = tr["t"], tr["V"], tr["q"], tr["profile"], tr["grid"]
    fam = families(n)
    vis = [v for v in meta["visits"] if v["replica"] == 0]
    for v in vis:
        v["fam"] = fam[tuple(v["klass"])]

    def bands(ax, lo, hi):
        for v in vis:
            if v["t_end"] >= lo and v["t_start"] <= hi:
                ax.axvspan(v["t_start"], max(v["t_end"], v["t_start"] + 0.06),
                           color=v["fam"]["color"], alpha=.55, lw=0)

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1.6, 1.7, 1.6, 2.2])
    ax = [fig.add_subplot(gs[i]) for i in range(5)]

    for k in range(n):
        ax[0].plot(t, q[:, k], lw=.35, color="0.45")
    bands(ax[0], t[0], t[-1])
    ax[0].set_ylabel("queen position")
    ax[0].set_title("whole run, coloured by solution family "
                    "(cool = modular, warm = non-modular)", fontsize=10)

    ax[1].plot(t, V, lw=.4, color="0.25")
    bands(ax[1], t[0], t[-1])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("V")
    ax[1].set_title("the witness", fontsize=10)

    # occupancy: one lane per family -- does it stay in a family or jump?
    for v in vis:
        f = v["fam"]
        ax[2].axvspan(v["t_start"], max(v["t_end"], v["t_start"] + 0.4),
                      ymin=f["idx"] / len(fam), ymax=(f["idx"] + 1) / len(fam),
                      color=f["color"], lw=0)
    ax[2].set_ylim(0, len(fam))
    ax[2].set_yticks([i + .5 for i in range(len(fam))],
                     [f"{d['label']} ({'mod' if d['modular'] else 'non'}, "
                      f"{d['size']})" for d in
                      sorted(fam.values(), key=lambda d: d["idx"])])
    ax[2].set_xlabel("time")
    ax[2].set_title("which family is on the board, when", fontsize=10)
    for a in ax[:3]:
        a.set_xlim(t[0], t[-1])
        a.axvspan(*window, facecolor="none", edgecolor="k", lw=1.2, ls=":")

    sel = (t >= window[0]) & (t <= window[1])
    ax[3].imshow(prof[sel].T, aspect="auto", origin="lower", cmap="magma",
                 extent=[window[0], window[1], grid[0], grid[-1]])
    ax[3].set_ylabel("column")
    ax[3].set_title(f"zoom {window[0]:.0f}-{window[1]:.0f}: superposition profile",
                    fontsize=10)

    for k in range(n):
        ax[4].plot(t[sel], q[sel][:, k], lw=1.0, color="0.35")
    bands(ax[4], *window)
    seen = {}
    for v in vis:
        if window[0] <= v["t_start"] <= window[1]:
            seen.setdefault(tuple(v["placement"]), v)
    for k, v in sorted(seen.items(), key=lambda kv: kv[1]["t_start"]):
        ax[4].annotate("".join(map(str, k)) + f"  {v['fam']['label']}",
                       (v["t_start"], n - 0.3), fontsize=7, rotation=90,
                       ha="center", va="top", color=v["fam"]["color"])
    for lev in range(n):
        ax[4].axhline(lev, color="0.9", lw=.5, zorder=0)
    ax[4].set_xlim(*window)
    ax[4].set_ylabel("queen position")
    ax[4].set_xlabel("time")
    ax[4].set_title("same window, waves grey, bands coloured by family",
                    fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=d["color"])
               for d in sorted(fam.values(), key=lambda d: d["idx"])]
    labels = [f"{d['label']}: {'modular' if d['modular'] else 'NON-modular'}, "
              f"{d['size']} solutions"
              for d in sorted(fam.values(), key=lambda d: d["idx"])]
    ax[0].legend(handles, labels, fontsize=8, ncol=3, loc="upper center")

    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    return fam, vis


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "winner_n7"
    meta = load(name)
    k = meta["kpi"]
    S = fig_waves(meta, f"{R}/{name}_waves.png")
    hops, dm, dn = fig_itinerary(meta, S, f"{R}/{name}_structure.png")
    fig_zoom(meta, f"{R}/{name}_zoom.png")
    fig_green(meta, f"{R}/{name}_green.png")
    fig_by_family(meta, f"{R}/{name}_families.png")

    print(f"{name}: {k['n_distinct']}/{k['n_solutions_total']} solutions, "
          f"{k['n_distinct_class']}/{k['n_classes_total']} classes, "
          f"{len(meta['visits'])} visits, {k['wall_s']:.0f}s")
    print(f"  modular found     : {len(k['solutions_modular'])} "
          f"(of 28)   total dwell {sum(dm):.0f}")
    print(f"  non-modular found : {len(k['solutions_non_modular'])} "
          f"(of 12)   total dwell {sum(dn):.0f}")
    if dm and dn:
        print(f"  median dwell: modular {np.median(dm):.2f} vs "
              f"non-modular {np.median(dn):.2f}")
    if hops:
        c = collections.Counter(hops)
        print(f"  hop sizes: {dict(sorted(c.items()))}  (min possible is 2)")
    print(f"\nwrote {R}/{name}_" + "{waves,structure,zoom,green,families}.png")
