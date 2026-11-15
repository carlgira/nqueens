"""The experiment grids.  Run as:  ./venv/bin/python -m wave_queens.experiments <name>"""

import sys

from .sweep import sweep, expand

R = "wave_queens/results"

GRIDS = {
    # Does a stiffer lattice term stop the standard potential from evading
    # conflicts by sliding off the integer grid?
    "diag_standard": (dict(
        mode=["standard"], n=[6], n_steps=[100_000], lam_probe=[0], seed=[1],
        temp=[0.2, 0.3, 0.45, 0.6], w_lat=[0.35, 1.0, 2.0], sigma=[0.30, 0.45]),
        f"{R}/diag_standard.jsonl"),

    # Standard mode, take 2: the Fourier column term is in place, now find the
    # weights.  Only ratios matter, so w_col is pinned to 1 and we scan the rest.
    "standard_grid": (dict(
        mode=["standard"], n=[6, 7], n_steps=[150_000], lam_probe=[0], seed=[1],
        temp=[0.10, 0.15, 0.22, 0.32, 0.45], w_diag=[0.5, 1.0, 2.0],
        w_lat=[0.10, 0.25, 0.50]),
        f"{R}/standard_grid.jsonl"),

    # Stage A of the standard-mode search.  The first standard_grid left tau,
    # sigma and the seed count unscanned; tau (the thermostat coupling) is the
    # most likely mixing knob and had been pinned at 1.0 the whole time.
    # Scored on distinct solutions, averaged over 3 seeds.
    "std_A": (dict(
        mode=["standard"], n=[6, 7], n_steps=[120_000], lam_probe=[0], w_diag=[1.0],
        seed=[1, 2, 3], temp=[0.10, 0.15, 0.22, 0.32], tau=[0.5, 1.0, 2.0, 4.0],
        sigma=[0.22, 0.30, 0.40], w_lat=[0.15, 0.25]),
        f"{R}/std_A.jsonl"),

    # Stage B: stage A saturated at 1-2 solutions no matter the parameters, and
    # n_perms never left single digits -- one temperature cannot be both hot
    # enough to rearrange queens and cold enough to rest in a solution.  Give it
    # a ladder instead and scan the ladder's own parameters.
    "std_B_pt": (dict(
        mode=["standard"], n=[6, 7], n_steps=[80_000], w_diag=[1.0], w_lat=[0.15],
        sigma=[0.40], tau=[2.0], seed=[1, 2],
        n_replicas=[4, 8], temp_min=[0.06, 0.12], temp_max=[0.4, 0.8, 1.6],
        swap_every=[100, 500]),
        f"{R}/std_B_pt.jsonl"),

    # Stage A hit the edge of its own grid on temp, tau and sigma.  Close that
    # methodological flank: same single-temperature model, ranges extended past
    # where stage A stopped.  If the ceiling is real it should hold out here too.
    "std_A2": (dict(
        mode=["standard"], n=[6, 7], n_steps=[120_000], lam_probe=[0], w_diag=[1.0],
        seed=[1, 2, 3], temp=[0.32, 0.45, 0.62], tau=[0.15, 0.3, 0.5],
        sigma=[0.40, 0.50], w_lat=[0.15, 0.25]),
        f"{R}/std_A2.jsonl"),

    # Stage C: n_replicas=8 won every top cell in stage B and was the largest
    # value scanned, so push the ladder further and run longer.
    "std_C_pt": (dict(
        mode=["standard"], n=[6, 7], n_steps=[120_000], w_diag=[1.0], w_lat=[0.15],
        sigma=[0.40], tau=[2.0], seed=[1, 2], swap_every=[100],
        n_replicas=[8, 16], temp_min=[0.06, 0.12], temp_max=[0.4, 1.6]),
        f"{R}/std_C_pt.jsonl"),

    # Stage D (defined, not run): stage C still had n_replicas and temp_max at
    # the edge of the grid and showed no sign of saturating, so push both, and
    # add N=8..12 to ask whether the ladder scales with N or the ceiling returns.
    # Expensive -- start with n=[8] alone if you want a quick read.
    "std_D_pt": (dict(
        mode=["standard"], n=[7, 8, 10, 12], n_steps=[200_000], w_diag=[1.0],
        w_lat=[0.15], sigma=[0.40], tau=[2.0], seed=[1, 2], swap_every=[100],
        n_replicas=[16, 32, 48], temp_min=[0.12], temp_max=[1.6, 3.0]),
        f"{R}/std_D_pt.jsonl"),

    # N=11 with the column term renormalised so w_col is the energy per column
    # collision at every N.  The N=7 equivalent is w_col = 2*1.0/7 = 0.29; the
    # scout at that strength left 2.2 queens per frame sharing a column (N=7 has
    # 1.0) and found nothing, because there are (N-1)/2 times more pairs able to
    # collide.  Scan up from there to find where mixing comes back.
    "n11_wcol": (dict(
        mode=["standard"], n=[11], col_norm=["pairs"], n_steps=[150_000],
        w_diag=[1.0], w_lat=[0.15], sigma=[0.40], tau=[2.0], swap_every=[100],
        n_replicas=[20], temp_min=[0.12], temp_max=[1.6], seed=[1, 2],
        w_col=[0.29, 0.6, 1.2, 2.4, 4.8]),
        f"{R}/n11_wcol.jsonl"),

    # w_col has to do two opposite jobs, the same trap as the single-temperature
    # ceiling: it must be big enough that queens stop sharing columns, but two
    # queens can only SWAP columns by passing through one, so a big w_col also
    # walls off the rearrangement mechanism.  At N=7 w_col(per collision)=0.29
    # against temp_max=1.6, a ratio of 5.6.  At N=11 with w_col=4.8 the ladder
    # tops out at 1.6 -- nowhere near enough to cross a 4.8 barrier.  So the top
    # of the ladder has to scale WITH w_col.  Read this grid along the diagonal.
    "n11_wcol_tmax": (dict(
        mode=["standard"], n=[11], col_norm=["pairs"], n_steps=[150_000],
        w_diag=[1.0], w_lat=[0.15], sigma=[0.40], tau=[2.0], swap_every=[100],
        n_replicas=[32], temp_min=[0.12], seed=[1, 2],
        w_col=[1.2, 2.4, 4.8], temp_max=[1.6, 7.0, 27.0]),
        f"{R}/n11_wcol_tmax.jsonl"),

    # The "temp_max must scale with w_col" idea is refuted: n11_wcol_tmax is best
    # at the LOWEST top (1.6) and gets worse as the ladder is stretched, because
    # stretching it spends rungs on useless hot replicas.  swap_rate 0.86 is also
    # far above the ~0.3 that PT wants, so the ladder is too dense.  So: keep the
    # range narrow, push w_col further (still rising at the edge), and spend the
    # freed replicas on nothing -- just check whether fewer rungs cost anything.
    "n11_wcol_deep": (dict(
        mode=["standard"], n=[11], col_norm=["pairs"], n_steps=[150_000],
        w_diag=[1.0], w_lat=[0.15], sigma=[0.40], tau=[2.0], swap_every=[100],
        temp_min=[0.12], seed=[1, 2], n_replicas=[12, 20],
        w_col=[4.8, 9.6, 19.2], temp_max=[1.6, 3.0]),
        f"{R}/n11_wcol_deep.jsonl"),

    # n11_wcol_deep decouples two things that had been moving together:
    #   w_col 4.8 -> 9.6 -> 19.2 :  permutations visited 22 -> 52 -> 91  (rises)
    #                               solutions per permutation 1/55 -> 1/130 -> 0
    # Throughput improves and the hit rate collapses, because w_diag stayed at
    # 1.0 while w_col went to 19.2 -- the diagonal term, which is the only thing
    # that distinguishes a solution from any other permutation, became a rounding
    # error.  The N=7 winner had w_diag/w_col = 1.0/0.286 = 3.5.  Restore that
    # ratio at high w_col and both should hold at once.
    "n11_ratio": (dict(
        mode=["standard"], n=[11], col_norm=["pairs"], n_steps=[150_000],
        w_lat=[0.15], sigma=[0.40], tau=[2.0], swap_every=[100], temp_min=[0.12],
        temp_max=[1.6], n_replicas=[20], seed=[1, 2],
        w_col=[9.6, 19.2], w_diag=[1.0, 4.0, 16.0, 67.0]),
        f"{R}/n11_ratio.jsonl"),

    # The "closed window" call was premature: the 68 N=11 runs varied 4 knobs and
    # pinned 5.  Two of the pinned ones are prime suspects -- sigma was the second
    # most influential knob at N=7 and finished at the edge of its grid there
    # (0.40 rising, and A2 preferred 0.50), and temp_min was 0.12 in every single
    # N=11 run even though the cold end is where solutions crystallise.  Run
    # length also matters: N=7's 18 solutions came from 300k steps, not 150k.
    # Same standard as std_A2: close the flank before declaring a limit.
    "n11_broad": (dict(
        mode=["standard"], n=[11], col_norm=["pairs"], n_steps=[300_000],
        w_diag=[1.0], w_lat=[0.15], tau=[2.0], swap_every=[100],
        n_replicas=[20], temp_max=[1.6], seed=[1, 2],
        w_col=[4.8, 9.6], sigma=[0.30, 0.40, 0.50, 0.60],
        temp_min=[0.05, 0.12, 0.25]),
        f"{R}/n11_broad.jsonl"),

    # N = 8, 9, 10.  Every attempt to PREDICT the parameter scaling from theory
    # failed (temp_max ~ w_col, w_diag/w_col ratio) while only the intensive
    # arguments held.  So derive it empirically instead: scan the three knobs
    # that actually moved N=11 (w_col, sigma, temp_min) at each N, and see
    # whether the best point drifts systematically.  With N=7..10 that is four
    # points to fit, and N=11 already has ~120 runs sitting there as a test set.
    # These N are also cleaner for the non-modular question: gcd(N,6) != 1 for
    # 8, 9 and 10, so they have no modular solutions at all.
    "scale_8910": (dict(
        mode=["standard"], n=[8, 9, 10], col_norm=["pairs"], n_steps=[200_000],
        w_diag=[1.0], w_lat=[0.15], tau=[2.0], swap_every=[100],
        n_replicas=[20], temp_max=[1.6], seed=[1, 2],
        w_col=[4.8, 9.6], sigma=[0.40, 0.50, 0.60], temp_min=[0.12, 0.25]),
        f"{R}/scale_8910.jsonl"),

    # Where is the intermittent band in temperature?
    "temp_scan": (dict(
        mode=["modular"], n=[7], n_steps=[200_000], lam_probe=[0], seed=[1, 2, 3],
        temp=[0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.32, 0.40]),
        f"{R}/temp_scan.jsonl"),
}


def main(name):
    grid, out = GRIDS[name]
    sweep(expand(grid), out)


if __name__ == "__main__":
    main(sys.argv[1])
