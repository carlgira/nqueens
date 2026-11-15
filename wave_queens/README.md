# wave_queens

N-queens as N coupled oscillators ("waves") whose superposition is flat exactly
on solutions.  The question being tested: **does the flatness signal alternate
between broadband noise and flat plateaus, with the plateaus being solutions,
and does the orbit walk from one solution to another?**

## The setup

Queen `k` is one degree of freedom `q_k`.  A potential `V(q) >= 0` vanishes
(modular) or bottoms out at a small `V*` (standard) exactly on solutions, so
`V(t)` is the *flatness witness*.

- **modular** — `q` = phases on the circle.
  `V = sum_{m=1..N-1} w_c|R_m|^2 + w_d(|D_m|^2 + |A_m|^2)` with `R/D/A` the DFT
  of the column / diagonal / antidiagonal profiles.  `V = 0` iff the phases are
  a rotated N-th root lattice in all three directions — a toroidal n-queens
  solution.  Self-discretising.  Only has zeros when `gcd(N,6) = 1`.
- **standard** — `q` = real column positions.  Pairwise Gaussian conflict bumps
  on columns and both diagonals + a lattice term + a box.  Works for all N.

## Why a Nose-Hoover chain and not a conservative Hamiltonian

The first version was conservative (plain Verlet, `E` exactly constant).  It
does not work, for a reason worth keeping: along a Hamiltonian orbit the time
density goes as `1 / sqrt(E - V)`, so the system moves **fastest** precisely
where `V = 0`.  It sprints through the solutions instead of resting in them.
Measured: `V_min ~ 0.8` over the whole run, essentially no flat plateaus.

A **per-queen Nose-Hoover chain** fixes both requirements at once:

- deterministic and chaotic — no random forcing, `lambda_max > 0`;
- samples `exp(-V/T)`, so the orbit genuinely dwells in the `V ~ 0` valleys and
  still escapes over the barriers — the intermittency we are looking for;
- one chain per queen pins `<p_k^2> = T` for every `k` individually, so every
  wave carries the same energy **by construction**.  `PR` (participation ratio
  of the time-averaged per-queen kinetic energy) measures it: `PR = 1` means
  perfectly democratic, `PR = 1/N` means one queen hogs everything.  Observed
  `PR > 0.99`.

## Files

- `core.py` — potentials, Nose-Hoover walker, measurement, brute-force ground
  truth, symmetry canonicalisation, `run(cfg) -> kpi`.
- `sweep.py` — append-only JSONL harness.  Run id = hash of config, so grids
  are resumable and parallel.
- `experiments.py` — the grids.  `python -m wave_queens.experiments <name>`
- `figures.py` — featured runs and phase diagrams.
- `plots.py` — the four-panel run figure, dwell statistics, phase diagrams.

## KPIs

| key | meaning |
|---|---|
| `flat_frac` | fraction of time an exact solution is on the board |
| `n_distinct` / `n_solutions_total` | solutions visited vs. brute-force truth |
| `dwell_mean`, `dwell_max` | how long a plateau lasts |
| `V_min` | closest approach to the flat state (relative to `V*`) |
| `PR` | is the energy shared equally among the queens |
| `lyapunov` | is the dynamics actually chaotic |
| `n_configs`, `n_perms` | mixing — is it exploring or stuck |
| `distinct_per_cpu_s` | the only number that compares parameter sets |

## Results so far

Ground truth (brute force): N=6 standard 4 solutions / 1 symmetry class, no
modular solutions at all (`gcd(6,6)=6`).  N=7 standard 40 / 6 classes, modular
28 / **1 class**.

**Modular N=7, T=0.11** (`results/modular_n7.png`) — the hypothesis holds.
71% of the time an exact solution is on the board; 255 visits; 14 of the 28
solutions; mean dwell 5.5, max 22.4; `lambda = 0.35 > 0`; `PR = 0.999`.
The witness distribution is visibly bimodal — a peak at the thermal floor
(flat phase) and a shoulder near `V ~ 1.2` (noisy phase).

**Temperature is the single control** (`results/temp_curve.png`).  The
intermittent band is `T ~ 0.09-0.13`.  Below it the orbit sticks in one valley
(long dwells, few distinct solutions, `n_perms ~ 3`); above it there is no
dwelling at all (`flat_frac -> 0`).

**Standard N=6 and N=7** (`results/standard_n6.png`, `standard_n7.png`) — the
noise/flat alternation is there (13% and 4% of the time flat, dwells up to 92
and 63) but the orbit does **not** walk: it locks into one solution and only
ever finds that one, `n_perms = 2-3`.  The bottleneck is mixing, and the cause
is structural: the standard diagonals cannot be written in Fourier, so that
term stays short-range and the landscape keeps large featureless plateaus.  A
Lorentzian (heavy-tailed) diagonal kernel was tried as a fix and is not better.

## Standard mode: the search for parameters that yield more solutions

Four sweeps, ~920 runs, all in `results/*.jsonl` and resumable (`run_id` is a
hash of the config, so re-launching a grid only fills the holes).

| stage | model | grid | N=6 best cell | N=7 best cell |
|---|---|---|---|---|
| A  | 1 temperature | 576 runs, temp x tau x sigma x w_lat, 3 seeds | 1.00 / 4 | 1.67 / 40 |
| A2 | 1 temperature, ranges extended past A's edges | 216 runs | 0.67 / 4 | 2.00 / 40 |
| B  | parallel tempering, 4-8 replicas | 96 runs | 3.00 / 4 | 3.50 / 40 |
| C  | parallel tempering, 8-16 replicas | 32 runs | **4.00 / 4** | **10.50 / 40** |

**A single temperature has a hard ceiling of 1-2 solutions.** Stage A hit the
edge of its own grid on temp, tau and sigma, so A2 re-ran with the ranges
extended past those edges; the ceiling held (N=7 went 1.67 -> 2.00).  Across
all 792 single-temperature runs `n_perms` never exceeded 14 out of N!.  The
cause is not tuning: one temperature cannot be simultaneously hot enough to
rearrange the queens and cold enough to rest in a solution.

**A temperature ladder breaks it and does not saturate.**  1 -> 8 -> 16
replicas gives 1.67 -> 3.5 -> 10.5 distinct solutions at N=7, with `n_perms`
going 12 -> 65 -> 158.  `n_replicas` is the only knob with a strong consistent
slope in the marginals; `temp_min`, `temp_max` and `swap_every` are nearly
flat.  The winner also has the best `distinct_per_cpu_s` in its table (0.0295),
so this is better use of CPU, not just more of it:

    n_replicas=16, temp_min=0.12, temp_max=1.6, swap_every=100,
    tau=2.0, sigma=0.40, w_lat=0.15, w_diag=1.0

Both `n_replicas` and `temp_max` were still at the edge of stage C's grid.
`std_D_pt` in `experiments.py` is defined but not run: it pushes them further
and adds N=8..12 to ask whether the ladder scales with N.

## Structure in the waves (featured run `winner_n7`, N=7, 300k steps)

Stored once by `featured.py` (config + every KPI + the solution list + the full
visit itinerary + traces) and re-read by `structure.py` without simulating.
**18 of 40 solutions, all 6 symmetry classes, 1334 visits, 277 s.**

**The waves have two phases** (`winner_n7_zoom.png`).  Outside a solution the
queens square-wave between integer columns, several hops per time unit.  Inside
one, all N lock onto distinct integers and only ripple gently; the superposition
profile becomes N clean parallel stripes and V drops by 3-4x.  The alternation
the hypothesis predicted is real and visible.

**The orbit does not walk between neighbouring solutions.**  Of 541 transitions,
397 move 6 of the 7 queens and 113 move all 7; only 2 move as few as 3.  It
dissolves and recrystallises somewhere unrelated rather than stepping.  This is
the part of the original hypothesis that is wrong.

**Residence is time-localised and non-recurrent.**  A solution is occupied
within a window of ~79 of 1200 time units and then essentially abandoned -- the
itinerary is a staircase, not a random walk over a fixed set.

**Dwell times are quantised, not exponential.**  The peaks are 0.38, 0.78, 1.18,
1.58, 2.38: an arithmetic comb of step ~0.40.  Memoryless thermal escape would
give an exponential distribution; a comb means escape is gated by the phase of
the local vibration.  The dynamics leaves a solution deterministically, at
specific phases, not by a random kick.

**No modular bias.**  13 of the 28 modular solutions (46%) and 5 of the 12
non-modular ones (42%) were found; median dwell is 0.38 for both.  Nothing in
the physics privileges the algebraically-constructible solutions -- which also
settles the earlier class-count bound of "2 to 8 non-modular" at 5.

## N=11: the soft potential's window closes

Parameters were extrapolated from N=7 by asking which quantities are intensive.
Three predictions, checked against 68 runs:

| prediction | outcome |
|---|---|
| T transfers unchanged (NHC is per-queen, so T is intensive) | **confirmed** -- kinetic energy per queen 0.059 (N=7) vs 0.060 (N=11), V per queen 0.108 vs 0.118, PR 0.999 both |
| `n_replicas` ~ sqrt(N) to hold swap acceptance | **confirmed** -- swap_rate 0.80 vs 0.81 |
| `w_col` ~ N | **partly** -- right direction, too small: it equalised the energy per collision but not the (N-1)/2 entropy of pairs able to collide |
| `temp_max` ~ `w_col` | **refuted** -- best at the *lowest* top; stretching the ladder spends rungs on useless hot replicas |
| `w_diag/w_col` = 3.5 (the N=7 value) | **refuted** -- raising w_diag collapses permutation throughput 78 -> 10 -> 0 |

`sum_{m=1}^{N-1}|S_m|^2 = 2NC` exactly, with C the number of pairs sharing a
column, so the old 1/N^2 normalisation made the energy per collision 2*w_col/N.
`col_norm="pairs"` normalises by 2N instead, so `w_col` is the per-collision
energy at every N (verified to 1e-14).  Default stays `"n2"` for run_id
stability.

**The result is a closed window** (`results/n11_tradeoff.png`).  Every parameter
direction buys throughput (distinct permutations visited) and pays in precision
(solutions per permutation), or the reverse:

| knob raised | throughput | precision |
|---|---|---|
| `w_col` | 22 -> 52 -> 91 | 1/55 -> 1/130 -> 0 |
| `w_diag` | 78 -> 10 -> 0 | improves, then nothing to be precise about |
| `temp_max` | falls | falls |

Best over all 68 runs: **3 distinct solutions of 2680**, 20 runs of 68 found any.
The N=7 operating point (118 permutations *and* a 1/13 hit rate at matched
steps) sits outside the region N=11 can reach at all.

Worth keeping: the hit rate per permutation only degrades 1/13 -> 1/55, not the
118x that the solution density would suggest.  The diagonal potential is still
doing real work (251x better than sampling permutations at random).  What
strangles N=11 is throughput, and `w_col` is the only thing that opens it --
while being the same term that has to stay large to keep queens off each other.
One knob, two opposite jobs: the same trap as the single-temperature ceiling.

## Formula space: do chaotic maps find solutions by chance?

`formulas.py` turns 12 families into permutations and measures the hit rate
against the only honest null, the density Q(N)/N!.  A uniform random permutation
is included and comes out at 1.0x, which validates the metric.

**No chaotic family beats chance.**  Raw enrichment is misleading -- `logistic`
looks like 9.4x at N=8 but emits only 296 distinct permutations from 200k
samples, so it wins by concentration, not alignment.  Corrected for repertoire
size (fraction of the family's OWN distinct output that is a solution), every
chaotic family lands between 1x and 4x.  The cat map and the Chirikov standard
map -- the two canonical chaotic systems -- sit at exactly 1.0x while emitting
>100k distinct permutations: uniform samplers with extra steps.  Chaos gives
coverage, not targeting; that is what mixing means.

**The winner is a zero-entropy family**, the rotation / Sturmian one:
`c_k = rank of frac(alpha*k + beta)`.  Its repertoire is finite and small, so
`rotation.py` enumerates it exactly (the permutation is constant on cells
bounded by Farey fractions of order N-1).  Every hit classifies into exactly
one of two mechanisms, with nothing left over:

| N | repertoire | hits | 1 step (modular) | 2 adjacent steps (Sturmian) | vs density |
|---|---|---|---|---|---|
| 7 | 84 | 28 | 28 | 0 | 42x |
| 8 | 144 | 0 | 0 | 0 | 0 |
| 9 | 198 | 6 | 0 | **6** | 31x |
| 10 | 280 | 12 | 0 | **12** | 215x |
| 11 | 352 | 90 | 88 | 2 | 3,808x |
| 12 | 504 | 10 | 0 | **10** | 669x |
| 13 | 598 | 138 | 130 | 8 | 19,495x |
| 14 | 812 | 0 | 0 | 0 | 0 |

Constant step is the classical mod-N construction, which needs `gcd(N,6)=1`.
Two *adjacent* step values is the three-distance / Sturmian signature, and it is
strictly more general: at N=9, 10 and 12 the classical construction does not
exist and every solution the family finds is Sturmian.  N=9 is the sharpest case
-- `gcd(9,6)=3` and `gcd(10,6)=2`, so neither the direct nor the truncated
mod-(N+1) construction applies, yet the family still yields 6 solutions:

    (1,7,4,2,8,5,3,0,6)  steps {6,7}      (3,6,8,1,4,7,0,2,5)  steps {2,3}
    (2,8,5,3,0,6,4,1,7)  steps {6,7}      (6,0,3,5,8,2,4,7,1)  steps {2,3}
    (5,2,0,7,4,1,8,6,3)  steps {6,7}      (7,1,4,6,0,3,5,8,2)  steps {2,3}

Still uncovered: N=8 and N=14 get nothing from this family (both have
`gcd(N,6) != 1` and `gcd(N+1,6) != 1`).

## Three closed forms, one per class of N mod 6

Extending the exact rotation enumeration to N=7..26 shows the residue class is
the invariant, and one class is invisible to the family:

| N mod 6 | rotation hits | why |
|---|---|---|
| 1, 5 | many, constant step | `gcd(N,6)=1`, the classical modular rotation |
| 0, 4 | Sturmian only | modular on N+1, truncated |
| 3 | Sturmian only | two-block word |
| **2** | **0 at N=8,14,20,26** | needs two step values separated by 3 |

The rotation family can only emit *adjacent* step values -- that is what the
three-distance theorem gives -- so N = 2 mod 6 is out of reach.  Its simplest
solutions have steps {2,5} (gap 3) with a single defect at the exact midpoint.

Working backwards from the winning parameters gives `constructions.py`, three
closed forms covering every N with no gaps (verified 4..600):

    gcd(N,6)=1   c_k = a*k mod N                       constant step
    N even       step 2, one defect at d=(N-2)/2       q=5,c0=3 if N=2 mod 6
                                                       q=1,c0=1 otherwise
    N = 3 mod 6  steps {N-3,N-2}, word 00 1^m 00 1^m 00, m=(N-7)/2, c0=1

**These formulas are not new** -- this is the classical explicit construction
(Hoffman-Loessi-Moore 1969, Bernhardsson 1991), which is likewise case-split on
N mod 6.  What the search contributed is deriving the case structure from data
instead of assuming it, and showing why the split is at 6.

**Bug found along the way:** translating all columns mod N preserves solutions
on the *torus* but not in standard n-queens, where wrapping changes `c_k +/- k`
as integers.  `twostep.py` fixed `c_0 = 0` on that false symmetry and reported
zeros; scanning `c_0` is required.

**Caveat on the walk.** All 28 modular N=7 solutions are a single symmetry
class, so the observed itinerary — a clean cyclic staircase through 7 states —
is largely the zero mode (the global rotation of the splay lattice) carrying
the configuration through its own translates.  Genuine transitions between
inequivalent solutions cannot be demonstrated at N=7 modular.  N=11 or N=13
(`gcd(N,6)=1`, many classes) is the test that would settle it.
