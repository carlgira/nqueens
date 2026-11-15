"""
Wave queens: N-queens as a conservative Hamiltonian system of N oscillators.

Each queen k is one "wave". A potential V(q) >= 0 vanishes exactly on
solutions, so V(t) is the flatness witness: while the orbit wanders the
signal is broadband noise, and every time it falls into a solution valley
the superposition goes flat for a while (chaotic itinerancy).

Two potentials share one integrator:

  modular  -- exact Fourier form.  q = phases theta_k on the circle,
              V = sum_{m=1}^{N-1} w_c|R_m|^2 + w_d(|D_m|^2 + |A_m|^2)
              with R/D/A the DFT of the column / diagonal / antidiagonal
              profiles.  V = 0 iff the phases are a rotated N-th root
              lattice in all three directions, i.e. a toroidal n-queens
              solution.  Self-discretising, but only has zeros when
              gcd(N, 6) = 1.

  standard  -- q = real column positions x_k.  Pairwise Gaussian conflict
              bumps on columns and on both diagonals, plus a lattice term
              that snaps to integers and a box that confines to the board.
              Works for every N; V at a solution is a small positive
              residual V*, reported alongside.

Every queen is a unit-modulus oscillator (modular) or carries an identical
pairwise coupling (standard): no wave can go quiet and let the others do
the cancelling.  The participation ratio PR measures whether that holds.
"""

import functools
import hashlib
import json
import math

import numpy as np

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# potentials
# --------------------------------------------------------------------------

class ModularPotential:
    """Exact Fourier potential for toroidal n-queens.  State = phases."""

    mode = "modular"
    periodic = True

    def __init__(self, n, w_col=1.0, w_diag=1.0):
        self.n = n
        self.w_col = w_col
        self.w_diag = w_diag
        m = np.arange(1, n)
        k = np.arange(n)
        self.m = m
        # e^{i 2 pi m k / N} and its conjugate, the diagonal / antidiagonal shifts
        self.shift_d = np.exp(1j * TWO_PI * np.outer(m, k) / n)
        self.shift_a = np.conj(self.shift_d)
        self.norm = 1.0 / (n * n)

    def energy_grad(self, q):
        n = self.n
        ph = np.exp(1j * np.outer(self.m, q))          # (N-1, N)  e^{i m theta_j}
        pd = ph * self.shift_d
        pa = ph * self.shift_a
        S = ph.sum(axis=1)
        T = pd.sum(axis=1)
        U = pa.sum(axis=1)

        v = (self.w_col * np.abs(S) ** 2
             + self.w_diag * (np.abs(T) ** 2 + np.abs(U) ** 2)).sum() * self.norm

        # d|S_m|^2/dtheta_j = -2 m Im( e^{i m theta_j} conj(S_m) )
        mm = self.m[:, None]
        g = (self.w_col * np.imag(ph * np.conj(S)[:, None])
             + self.w_diag * (np.imag(pd * np.conj(T)[:, None])
                              + np.imag(pa * np.conj(U)[:, None])))
        grad = (-2.0 * mm * g).sum(axis=0) * self.norm
        return v, grad

    def columns(self, q):
        """Read out integer columns, de-rotating by the lattice-locking phase."""
        n = self.n
        lock = np.exp(1j * n * q).sum()
        phi = np.angle(lock) / n if abs(lock) > 1e-12 else 0.0
        return np.mod(np.rint((q - phi) * n / TWO_PI).astype(int), n)

    def profile(self, q, grid, kappa=0.18):
        """Smoothed column occupancy on `grid` (in column units), periodic."""
        d = (grid[:, None] - (q[None, :] * self.n / TWO_PI)) % self.n
        d = np.minimum(d, self.n - d)
        return np.exp(-0.5 * (d / kappa) ** 2).sum(axis=1)

    def random_state(self, rng):
        return rng.uniform(0.0, TWO_PI, self.n)

    def wrap(self, q):
        return np.mod(q, TWO_PI)

    def v_star(self):
        return 0.0


class StandardPotential:
    """Potential for ordinary (non-toroidal) n-queens.  State = column positions.

    Columns use the same Fourier splay term as the modular potential: it is the
    Dirichlet kernel, long-ranged, so there is a gradient everywhere pushing the
    queens towards equal spacing, and its barrier to two queens crossing is
    finite.  A short-range Gaussian column term was tried first and fails badly
    -- the landscape becomes flat plateaus with isolated bumps and the dynamics
    has nothing to follow (see results/diag_standard.jsonl).

    Diagonals cannot be Fourier here (they do not wrap), so they stay as pairwise
    Gaussian bumps at delta = +/-(j-k).  A lattice term pins the global rotation
    that the splay term leaves free, and a box keeps the queens on the board so
    the non-periodic diagonal term never sees a wrap-around.
    """

    mode = "standard"
    periodic = False

    def __init__(self, n, w_col=1.0, w_diag=2.0, w_lat=0.35, w_box=4.0, sigma=0.30,
                 kernel="gauss", col_norm="n2"):
        self.n = n
        self.w_col = w_col
        self.w_diag = w_diag
        self.w_lat = w_lat
        self.w_box = w_box
        self.sigma = sigma
        self.kernel = kernel
        # sum_{m=1}^{N-1} |S_m|^2 = 2*N*C exactly, with C the number of pairs of
        # queens sharing a column.  So the 1/N^2 normalisation ("n2") makes the
        # energy per collision 2*w_col/N -- it decays with N, which is why the
        # N=7 parameters starve the column constraint at N=11.  "pairs"
        # normalises by 2N instead, so w_col IS the energy per collision at
        # every N.  Default stays "n2" so stored run_ids remain comparable.
        self.col_scale = 1.0 / (2 * n) if col_norm == "pairs" else 1.0 / (n * n)
        self.col_norm = col_norm
        k = np.arange(n)
        self.m = np.arange(1, n)
        self.dk = (k[:, None] - k[None, :]).astype(float)   # row separations
        self.eye = np.eye(n, dtype=bool)

    def _bump(self, u):
        if self.kernel == "lorentz":
            # heavy tails: unlike a Gaussian this leaves a gradient everywhere,
            # so the orbit is never on a featureless plateau
            d = 1.0 / (1.0 + (u / self.sigma) ** 2)
            return d, -2.0 * u / self.sigma ** 2 * d * d
        g = np.exp(-0.5 * (u / self.sigma) ** 2)
        return g, -(u / self.sigma ** 2) * g

    def energy_grad(self, q):
        n = self.n
        delta = q[:, None] - q[None, :]

        gp, dgp = self._bump(delta - self.dk)
        gm, dgm = self._bump(delta + self.dk)
        for a in (gp, dgp, gm, dgm):
            a[self.eye] = 0.0

        v = 0.5 * self.w_diag * (gp.sum() + gm.sum())
        grad = self.w_diag * (dgp.sum(axis=1) + dgm.sum(axis=1))

        # columns: Fourier splay on theta = 2 pi x / N
        theta = TWO_PI * q / n
        ph = np.exp(1j * np.outer(self.m, theta))
        S = ph.sum(axis=1)
        v += self.w_col * (np.abs(S) ** 2).sum() * self.col_scale
        grad += (self.w_col * (TWO_PI / n) * self.col_scale) * (
            -2.0 * self.m[:, None] * np.imag(ph * np.conj(S)[:, None])).sum(axis=0)

        # lattice: pull onto integer columns (pins the rotation splay leaves free)
        v += self.w_lat * (1.0 - np.cos(TWO_PI * q)).sum()
        grad += self.w_lat * TWO_PI * np.sin(TWO_PI * q)

        # box: confine to [-0.5, n-0.5]
        hi = np.maximum(q - (n - 0.5), 0.0)
        lo = np.maximum(-0.5 - q, 0.0)
        v += self.w_box * (hi ** 2 + lo ** 2).sum()
        grad += self.w_box * 2.0 * (hi - lo)
        return v, grad

    def columns(self, q):
        return np.clip(np.rint(q).astype(int), 0, self.n - 1)

    def profile(self, q, grid, kappa=0.18):
        d = grid[:, None] - q[None, :]
        return np.exp(-0.5 * (d / kappa) ** 2).sum(axis=1)

    def random_state(self, rng):
        return rng.uniform(-0.5, self.n - 0.5, self.n)

    def wrap(self, q):
        return q

    def v_star(self):
        """Residual potential at an exact solution (soft bumps never reach 0)."""
        sols = enumerate_solutions(self.n, False, first_only=True)
        if not sols:
            return 0.0
        return self.energy_grad(np.array(sols[0], dtype=float))[0]


def make_potential(cfg):
    mode, n = cfg["mode"], cfg["n"]
    cls = ModularPotential if mode == "modular" else StandardPotential
    allowed = {"modular": ("w_col", "w_diag"),
               "standard": ("w_col", "w_diag", "w_lat", "w_box", "sigma", "kernel")}[mode]
    return cls(n, **{k: v for k, v in cfg.items() if k in allowed})


# --------------------------------------------------------------------------
# ground truth + symmetry canonicalisation
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def enumerate_solutions(n, modular, first_only=False):
    """Every solution, by backtracking, so the KPIs can say 'found X of Y'.

    Was a scan over all n! permutations, which is fine to n=8 and hopeless at
    n=11 (39.9M).  Backtracking gets n=11's 2680 solutions in well under a
    second.  `first_only` is for v_star(), which just needs one.
    """
    out, col = [], [0] * n
    used, d1, d2 = set(), set(), set()

    def place(r):
        if r == n:
            out.append(tuple(col))
            return first_only
        for c in range(n):
            if c in used:
                continue
            a, b = ((r + c) % n, (r - c) % n) if modular else (r + c, r - c)
            if a in d1 or b in d2:
                continue
            col[r] = c
            used.add(c), d1.add(a), d2.add(b)
            if place(r + 1):
                return True
            used.discard(c), d1.discard(a), d2.discard(b)
        return False

    place(0)
    return out


def canonical(c, n, modular):
    """Lexicographically smallest member of the symmetry orbit of a placement."""
    def flip_rows(a):
        return a[::-1]

    def flip_cols(a):
        return (n - 1 - a) % n if modular else n - 1 - a

    def transpose(a):
        return np.argsort(a)

    gens = [flip_rows, flip_cols, transpose]
    if modular:
        gens += [lambda a: (a + 1) % n, lambda a: np.roll(a, 1)]

    start = np.asarray(c, dtype=int)
    seen = {tuple(int(v) for v in start)}
    stack = [start]
    while stack:
        x = stack.pop()
        for g in gens:
            y = np.asarray(g(x), dtype=int)
            t = tuple(int(v) for v in y)
            if t not in seen:
                seen.add(t)
                stack.append(y)
    return min(seen)


def is_solution(c, n, modular):
    if len(set(c.tolist())) != n:
        return False
    if modular:
        return (len({(int(c[i]) + i) % n for i in range(n)}) == n
                and len({(int(c[i]) - i) % n for i in range(n)}) == n)
    return (len({int(c[i]) + i for i in range(n)}) == n
            and len({int(c[i]) - i for i in range(n)}) == n)


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------

class HillBias:
    """Metadynamics-style memory: once the orbit has sat in a solution long
    enough, drop a repulsive Gaussian hill on that configuration so it has to
    leave and go find a different one.  `w = 0` disables it entirely.

    The hill is a product of Gaussians over all N coordinates, so it only bites
    within ~sigma of that exact placement and leaves the rest of the landscape
    untouched.  The recorded witness V never includes the bias.
    """

    def __init__(self, n, w=0.0, sigma=0.5, after=2.0):
        self.n, self.w, self.sigma, self.after = n, w, sigma, after
        self.centers = np.zeros((0, n))

    @property
    def on(self):
        return self.w > 0.0

    def energy_grad(self, q):
        if not self.on or not len(self.centers):
            return 0.0, 0.0
        d = q[None, :] - self.centers
        e = np.exp(-0.5 * (d ** 2).sum(axis=1) / self.sigma ** 2)
        return (self.w * e.sum(),
                -self.w / self.sigma ** 2 * (e[:, None] * d).sum(axis=0))

    def deposit(self, c):
        self.centers = np.vstack([self.centers, np.asarray(c, float)])


class Walker:
    """One trajectory: positions, momenta and a per-queen Nose-Hoover chain.

    Plain Verlet (thermostat='none') conserves E exactly, but it is the wrong
    dynamics for this hypothesis: the time density along an orbit goes as
    1/sqrt(E - V), so the system moves *fastest* exactly where V = 0.  It
    sprints through the solutions instead of resting in them.

    A per-degree-of-freedom Nose-Hoover chain fixes both problems at once.  It
    is deterministic and chaotic (no random forcing), it samples exp(-V/T) so
    the orbit genuinely dwells in the V ~ 0 valleys and escapes over the
    barriers, and because the chain is attached to each queen separately it
    pins <p_k^2> = T for every k -- equal energy per wave by construction, not
    by luck.
    """

    def __init__(self, pot, q, p, dt, temp, tau, thermostat="nhc", bias=None):
        self.pot, self.dt = pot, dt
        self.q, self.p = q, p
        self.temp, self.thermostat = temp, thermostat
        self.bias = bias or HillBias(pot.n)
        self.Q1 = self.Q2 = max(temp, 1e-6) * tau * tau
        self.xi1 = np.zeros(pot.n)
        self.xi2 = np.zeros(pot.n)
        self._force(q)

    def _force(self, q):
        """Witness V is always the bare potential; only the force sees the bias."""
        self.v, g = self.pot.energy_grad(q)
        _, gb = self.bias.energy_grad(q)
        self.g = g + gb

    def _nhc_half(self):
        d = 0.5 * self.dt
        T, Q1, Q2 = self.temp, self.Q1, self.Q2
        g2 = (Q1 * self.xi1 ** 2 - T) / Q2      # chain link 2 is driven by link 1
        self.xi2 += 0.25 * d * g2
        for _ in range(2):
            self.xi1 *= np.exp(-0.125 * d * self.xi2)
            self.xi1 += 0.25 * d * (self.p ** 2 - T) / Q1
            self.xi1 *= np.exp(-0.125 * d * self.xi2)
            if _ == 0:
                self.p *= np.exp(-0.5 * d * self.xi1)
        g2 = (Q1 * self.xi1 ** 2 - T) / Q2
        self.xi2 += 0.25 * d * g2

    def step(self):
        if self.thermostat == "nhc":
            self._nhc_half()
        self.p -= 0.5 * self.dt * self.g
        self.q = self.pot.wrap(self.q + self.dt * self.p)
        self._force(self.q)
        self.p -= 0.5 * self.dt * self.g
        if self.thermostat == "nhc":
            self._nhc_half()

    def clone_perturbed(self, delta):
        w = object.__new__(Walker)
        w.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in self.__dict__.items()})
        w.q = self.pot.wrap(self.q + delta)
        w._force(w.q)
        return w


def init_state(pot, temp, rng):
    q = pot.random_state(rng)
    p = rng.normal(size=pot.n) * math.sqrt(max(temp, 1e-6))
    return q, p


def simulate(pot, walker, n_steps, record_every=20, modular=None, grid_pts=None,
             lam_probe=True, gap_tol=0.05, check_every=5):
    """Run the walker and record the witness signals + the solution itinerary."""
    n = pot.n
    dt = walker.dt
    modular = pot.mode == "modular" if modular is None else modular

    n_rec = n_steps // record_every + 1
    ts, vs, es = (np.empty(n_rec) for _ in range(3))
    qs = np.empty((n_rec, n))
    kin_sum = np.zeros(n)                    # time-averaged energy per queen
    grid = np.linspace(0, n, grid_pts if grid_pts else 8 * n, endpoint=False)
    profs = np.empty((n_rec, grid.size))

    d0 = 1e-9
    lam_sum, lam_cnt, lam_every = 0.0, 0, 25
    shadow = walker.clone_perturbed(d0 * np.eye(n)[0]) if lam_probe else None

    visits = []           # (t_start, t_end, raw placement, symmetry class)
    cur = None
    deposited = False
    flat_checks = n_checks = 0
    cache = {}
    seen_cfg, seen_perm = set(), set()
    r = 0
    for step in range(n_steps + 1):
        t = step * dt
        q, p = walker.q, walker.p
        if step % record_every == 0:
            kin_k = 0.5 * p ** 2
            ts[r] = t
            vs[r] = walker.v
            es[r] = walker.v + kin_k.sum()
            kin_sum += kin_k
            qs[r] = q
            profs[r] = pot.profile(q, grid)
            r += 1

        if step % check_every == 0:
            n_checks += 1
            key = tuple(int(x) for x in pot.columns(q))
            seen_cfg.add(key)
            ok = cache.get(key)
            if ok is None:
                arr = np.asarray(key)
                if len(set(key)) == n:
                    seen_perm.add(key)
                ok = canonical(arr, n, modular) if is_solution(arr, n, modular) else False
                cache[key] = ok
            elif len(set(key)) == n:
                seen_perm.add(key)
            if ok:
                flat_checks += 1
                if cur is not None and cur[2] == key and t - cur[1] <= gap_tol:
                    cur[1] = t
                    if (walker.bias.on and not deposited
                            and t - cur[0] > walker.bias.after):
                        walker.bias.deposit(key)
                        walker._force(walker.q)
                        deposited = True
                else:
                    if cur is not None:
                        visits.append(tuple(cur))
                    cur = [t, t, key, ok]
                    deposited = False
            elif cur is not None and t - cur[1] > gap_tol:
                visits.append(tuple(cur))
                cur = None

        if step == n_steps:
            break
        walker.step()

        if lam_probe:
            shadow.step()
            if step % lam_every == lam_every - 1:
                dq = shadow.q - walker.q
                if pot.periodic:
                    dq = (dq + math.pi) % TWO_PI - math.pi
                dp = shadow.p - walker.p
                dx1, dx2 = shadow.xi1 - walker.xi1, shadow.xi2 - walker.xi2
                d = math.sqrt(np.sum(dq ** 2) + np.sum(dp ** 2)
                              + np.sum(dx1 ** 2) + np.sum(dx2 ** 2))
                if d > 1e-15:
                    lam_sum += math.log(d / d0)
                    lam_cnt += 1
                    s = d0 / d
                    shadow.q = pot.wrap(walker.q + dq * s)
                    shadow.p = walker.p + dp * s
                    shadow.xi1 = walker.xi1 + dx1 * s
                    shadow.xi2 = walker.xi2 + dx2 * s
                    shadow.v, shadow.g = pot.energy_grad(shadow.q)

    if cur is not None:
        visits.append(tuple(cur))
    lam = lam_sum / (lam_cnt * lam_every * dt) if lam_cnt else float("nan")
    kin_avg = kin_sum / max(r, 1)
    pr = float(kin_avg.sum() ** 2 / (n * np.sum(kin_avg ** 2) + 1e-300))
    return dict(t=ts[:r], V=vs[:r], E=es[:r], q=qs[:r], profile=profs[:r], grid=grid,
                visits=visits, kin_avg=kin_avg, PR=pr, lyapunov=lam,
                flat_frac=flat_checks / max(n_checks, 1),
                n_configs=len(seen_cfg), n_perms=len(seen_perm))


# --------------------------------------------------------------------------
# one experiment = one config -> one row of KPIs
# --------------------------------------------------------------------------

class SolutionTracker:
    """Turns a stream of board readouts into dwell intervals on exact solutions."""

    def __init__(self, n, modular, gap_tol=0.05):
        self.n, self.modular, self.gap_tol = n, modular, gap_tol
        self.cache = {}
        self.visits = []
        self.cur = None
        self.deposited = False
        self.flat_checks = self.n_checks = 0
        self.seen_cfg, self.seen_perm = set(), set()

    def classify(self, key):
        ok = self.cache.get(key)
        if ok is None:
            arr = np.asarray(key)
            ok = (canonical(arr, self.n, self.modular)
                  if is_solution(arr, self.n, self.modular) else False)
            self.cache[key] = ok
        return ok

    def update(self, t, key):
        """Returns the canonical class if a fresh deposit is warranted."""
        self.n_checks += 1
        self.seen_cfg.add(key)
        if len(set(key)) == self.n:
            self.seen_perm.add(key)
        ok = self.classify(key)
        if ok:
            self.flat_checks += 1
            if self.cur is not None and self.cur[2] == key and t - self.cur[1] <= self.gap_tol:
                self.cur[1] = t
            else:
                if self.cur is not None:
                    self.visits.append(tuple(self.cur))
                self.cur = [t, t, key, ok]
                self.deposited = False
        elif self.cur is not None and t - self.cur[1] > self.gap_tol:
            self.visits.append(tuple(self.cur))
            self.cur = None
        return ok

    def close(self):
        if self.cur is not None:
            self.visits.append(tuple(self.cur))
            self.cur = None
        return self.visits


def run_pt(cfg, keep_traces=False):
    """Parallel tempering: M replicas on a temperature ladder, swapping states.

    The single-temperature runs are stuck on a mixing problem, not a parameter
    problem: over a 5-D grid `n_perms` never leaves single digits out of N!.
    One temperature cannot be both hot enough to rearrange the queens and cold
    enough to rest in a solution.  A ladder can: the hot replicas scramble, the
    cold ones settle, and the swaps carry configurations between them.  Every
    replica reports its own solutions, so the union is what we score.
    """
    c = dict(DEFAULTS)
    c.update(cfg)
    pot = make_potential(c)
    modular = c["mode"] == "modular"
    rng = np.random.default_rng(c["seed"])
    m = c["n_replicas"]
    temps = np.geomspace(c["temp_min"], c["temp_max"], m)

    walkers = []
    for temp in temps:
        q, p = init_state(pot, temp, rng)
        walkers.append(Walker(pot, q, p, c["dt"], temp, c["tau"], c["thermostat"]))
    trackers = [SolutionTracker(c["n"], modular) for _ in range(m)]

    dt, swap_every = c["dt"], c["swap_every"]
    check_every = 5
    n_rec = c["n_steps"] // c["record_every"] + 1
    ts, v_cold = np.empty(n_rec), np.empty(n_rec)
    kin_sum = np.zeros(c["n"])
    grid = np.linspace(0, c["n"], 8 * c["n"], endpoint=False)
    qs = np.empty((n_rec, c["n"])) if keep_traces else None
    profs = np.empty((n_rec, grid.size)) if keep_traces else None
    swaps_tried = swaps_done = 0
    r = 0

    for step in range(c["n_steps"] + 1):
        t = step * dt
        if step % c["record_every"] == 0:
            ts[r] = t
            v_cold[r] = walkers[0].v
            kin_sum += 0.5 * walkers[0].p ** 2
            if keep_traces:
                qs[r] = walkers[0].q
                profs[r] = pot.profile(walkers[0].q, grid)
            r += 1
        if step % check_every == 0:
            for w, tr in zip(walkers, trackers):
                tr.update(t, tuple(int(x) for x in pot.columns(w.q)))
        if step == c["n_steps"]:
            break
        for w in walkers:
            w.step()
        if swap_every and step % swap_every == swap_every - 1:
            # alternate even/odd adjacent pairs so every replica gets a chance
            start = (step // swap_every) % 2
            for i in range(start, m - 1, 2):
                a, b = walkers[i], walkers[i + 1]
                delta = (1.0 / a.temp - 1.0 / b.temp) * (a.v - b.v)
                swaps_tried += 1
                if delta >= 0 or rng.random() < math.exp(delta):
                    swaps_done += 1
                    a.q, b.q = b.q.copy(), a.q.copy()
                    a.xi1, b.xi1 = b.xi1.copy(), a.xi1.copy()
                    a.xi2, b.xi2 = b.xi2.copy(), a.xi2.copy()
                    sa = math.sqrt(a.temp / b.temp)
                    a.p, b.p = b.p * sa, a.p / sa
                    a._force(a.q)
                    b._force(b.q)

    all_visits, raw, classes = [], set(), set()
    for i, tr in enumerate(trackers):
        for v in tr.close():
            all_visits.append(v + (i,))
            raw.add(v[2])
            classes.add(v[3])
    all_visits.sort(key=lambda v: v[0])
    all_sols = enumerate_solutions(c["n"], modular)
    n_total_canon = len({canonical(np.array(s), c["n"], modular) for s in all_sols})
    dwells = np.array([v[1] - v[0] for v in all_visits]) if all_visits else np.zeros(0)
    kin_avg = kin_sum / max(r, 1)
    t_end = float(ts[r - 1])

    kpi = dict(
        run_id=run_id(c), **c,
        v_star=float(pot.v_star()),
        V_min=float(v_cold[:r].min() - pot.v_star()),
        V_mean=float(v_cold[:r].mean() - pot.v_star()),
        V_std=float(v_cold[:r].std()),
        kin_per_queen=float(kin_avg.mean()),
        PR=float(kin_avg.sum() ** 2 / (c["n"] * np.sum(kin_avg ** 2) + 1e-300)),
        lyapunov=float("nan"),
        flat_frac=float(trackers[0].flat_checks / max(trackers[0].n_checks, 1)),
        flat_frac_any=float(np.mean([tr.flat_checks / max(tr.n_checks, 1)
                                     for tr in trackers])),
        n_configs=len(set().union(*[tr.seen_cfg for tr in trackers])),
        n_perms=len(set().union(*[tr.seen_perm for tr in trackers])),
        n_visits=len(all_visits),
        n_distinct=len(raw),
        n_distinct_class=len(classes),
        n_solutions_total=len(all_sols),
        n_classes_total=n_total_canon,
        coverage=len(raw) / len(all_sols) if all_sols else 0.0,
        dwell_mean=float(dwells.mean()) if dwells.size else 0.0,
        dwell_max=float(dwells.max()) if dwells.size else 0.0,
        swap_rate=swaps_done / max(swaps_tried, 1),
        n_hills=0,
        # the placements themselves, not just how many -- so questions like
        # "were those modular?" can be answered from the JSONL without re-running
        solutions=sorted(list(k) for k in raw),
        t_end=t_end,
        visits_per_time=len(all_visits) / t_end if t_end else 0.0,
    )
    if not keep_traces:
        return kpi, None
    return kpi, dict(t=ts[:r], V=v_cold[:r], q=qs[:r], profile=profs[:r], grid=grid,
                     visits=all_visits, temps=temps)


DEFAULTS = dict(mode="standard", n=7, temp=0.35, tau=1.0, thermostat="nhc",
                dt=0.004, n_steps=250_000,
                w_col=1.0, w_diag=1.0, w_lat=0.35, w_box=4.0, sigma=0.30, kernel="gauss",
                col_norm="n2",
                bias_w=0.0, bias_sigma=0.5, bias_after=2.0,
                n_replicas=1, temp_min=0.10, temp_max=0.60, swap_every=250,
                seed=0, record_every=20, lam_probe=1)


def run_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run(cfg, keep_traces=False):
    c = dict(DEFAULTS)
    c.update(cfg)
    pot = make_potential(c)
    modular = c["mode"] == "modular"
    rng = np.random.default_rng(c["seed"])
    q, p = init_state(pot, c["temp"], rng)
    bias = HillBias(c["n"], c["bias_w"], c["bias_sigma"], c["bias_after"])
    walker = Walker(pot, q, p, c["dt"], c["temp"], c["tau"], c["thermostat"], bias)
    res = simulate(pot, walker, c["n_steps"], record_every=c["record_every"],
                   modular=modular, lam_probe=bool(c["lam_probe"]))

    all_sols = enumerate_solutions(c["n"], modular)
    total = len(all_sols)
    raw = {v[2] for v in res["visits"]}
    classes = {v[3] for v in res["visits"]}
    n_total_canon = len({canonical(np.array(s), c["n"], modular) for s in all_sols})
    vstar = pot.v_star()
    t_end = res["t"][-1]
    dwells = np.array([v[1] - v[0] for v in res["visits"]]) if res["visits"] else np.zeros(0)

    kpi = dict(
        run_id=run_id(c), **c,
        v_star=float(vstar),
        V_min=float(res["V"].min() - vstar),
        V_mean=float(res["V"].mean() - vstar),
        V_std=float(res["V"].std()),
        kin_per_queen=float(res["kin_avg"].mean()),
        PR=float(res["PR"]),
        n_configs=res["n_configs"],
        n_perms=res["n_perms"],
        lyapunov=float(res["lyapunov"]),
        flat_frac=float(res["flat_frac"]),
        n_visits=len(res["visits"]),
        n_distinct=len(raw),
        n_distinct_class=len(classes),
        n_solutions_total=total,
        n_classes_total=n_total_canon,
        coverage=len(raw) / total if total else 0.0,
        dwell_mean=float(dwells.mean()) if dwells.size else 0.0,
        dwell_max=float(dwells.max()) if dwells.size else 0.0,
        n_hills=int(len(walker.bias.centers)),
        solutions=sorted(list(k) for k in raw),
        t_end=float(t_end),
        visits_per_time=len(res["visits"]) / t_end if t_end else 0.0,
    )
    return (kpi, res) if keep_traces else (kpi, None)
