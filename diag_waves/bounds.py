"""
Bounds on the summed diagonal wave S(x).

Two facts drive everything:

1. The first three spectral moments are INVARIANT over every permutation:

	   M0 = sum_k c_k        = 2N
	   M1 = sum_k c_k * k    = 4N^2
	   M2 = sum_k c_k * k^2  = (2/3)N(N-1)(2N-1) + 2N(N-1) + N + 9N^3

   S is even, so its Taylor head  S(x) = M0 - M2 x^2/2 + M4 x^4/24 - ...
   is IDENTICAL for every permutation through the x^2 term. That is why all
   the curves leave x=0 together. M4 is the first moment that varies.

2. Solutions additionally have c in {0,1}. So the spectrum of a solution is a
   pair of N-subsets (A, B) with A subset of [1,2N-1], B = 14 + b, and

	   sum(A) = sum(b) = N^2          (from M1, split per band)
	   sum(A^2) + sum(b^2) = m2       (from M2)

   Enumerating the (A, b) pairs that satisfy those three linear/quadratic
   constraints gives a rigorous OUTER envelope for every solution -- computed
   from N alone, without ever solving the N-queens problem.
"""

import numpy as np
from itertools import combinations, permutations
from collections import defaultdict

from diag_waves.diag_waves import band_series, channels, shift, conflicts


def moments(n):
	"""The three invariant moments, in closed form."""
	m0 = 2 * n
	m1 = 4 * n * n
	m2 = (2 / 3) * n * (n - 1) * (2 * n - 1) + 2 * n * (n - 1) + n + 9 * n ** 3
	return m0, m1, int(round(m2))


def band_target(n):
	"""sum(A^2) + sum(b^2) for the two UNSHIFTED bands (b = k - 2N)."""
	_, _, m2 = moments(n)
	return m2 - (4 * n ** 2 * n + 4 * n * n * n)		# strip the 2N offset of band B


def feasible_pairs(n):
	"""All (A, b) obeying the three moment constraints. A, b subsets of [1,2N-1]."""
	cand = range(1, 2 * n)
	subs = [A for A in combinations(cand, n) if sum(A) == n * n]
	by_sq = defaultdict(list)
	for A in subs:
		by_sq[sum(a * a for a in A)].append(A)
	tgt = band_target(n)
	pairs = []
	for s, group in by_sq.items():
		for B in by_sq.get(tgt - s, ()):
			pairs.extend((A, B) for A in group)
	return pairs


def pair_wave(pair, n, x):
	A, b = pair
	k = np.array(list(A) + [2 * n + v for v in b])
	return np.cos(np.outer(k, x)).sum(axis=0)


def envelope(waves):
	w = np.asarray(waves)
	return w.min(axis=0), w.max(axis=0)


def all_perm_waves(n, x):
	basis = np.cos(np.outer(np.arange(4 * n), x))
	sol, other = [], []
	for p in permutations(range(n)):
		dp, dm = channels(p)
		y = band_series(p) @ basis
		(sol if conflicts(dp) == 0 and conflicts(dm) == 0 else other).append(y)
	return np.array(sol), np.array(other)


def taylor_head(n, x, order=2):
	"""The universal head: same for every permutation."""
	m0, _, m2 = moments(n)
	y = m0 - m2 * x ** 2 / 2
	return y


def modular_value(coef, m, j=1):
	"""
	S(2*pi*j/m) depends ONLY on the residue profile of the occupied harmonics:

		S(2*pi*j/m) = sum_r n_r * cos(2*pi*j*r/m),   n_r = #harmonics = r (mod m)

	So at every rational multiple of pi the wave collapses to a modular count and
	can only land on a lattice. Closed forms (j=1), verified to 1e-14:

		m=2:  n_0 - n_1                     (par menos impar)
		m=3:  (3/2)*n_0 - N
		m=4:  n_0 - n_2
		m=6:  (n_0 - n_3) + (n_1 + n_5 - n_2 - n_4)/2

	The single universal invariant, found by exhaustive search over m = 2..18 and
	every j coprime to m, is

		N even  =>  S(pi) = 0   for EVERY permutation

	For N odd, S(pi) = 2*(N - 2e) with e = #{i : i + perm[i] even}; parity forces
	e = N (mod 2), so the lattice step is 8, not 4.
	"""
	k = np.arange(len(coef))
	n_r = np.array([coef[k % m == r].sum() for r in range(m)])
	return float(n_r @ np.cos(2 * np.pi * j * np.arange(m) / m))


def parity_split(pair):
	"""
	#odd diagonals occupied in the d+ band. Equals 2t, where t counts even rows
	mapped to even columns, because a+ = i + perm[i] + 1 is odd exactly when i
	and perm[i] share parity.

	Solutions confine this to a narrow band around N/2 while the moment
	constraints happily allow the extremes -- the one genuinely NEW restriction
	found beyond M0/M1/M2. Measured:

		N=7  soluciones [3,5]      factibles [1,3,5]
		N=9  soluciones [3,5]      factibles [3,5,7]
		N=10 soluciones [4,6]      factibles [2,4,6,8]
		N=11 soluciones [3,5,7]    factibles [1,3,5,7,9]

	Worth noting what does NOT help: the per-band second moment sum(A^2) has an
	identical range over solutions and over the feasible set, for every N tested.
	"""
	A, _ = pair
	return sum(1 for a in A if a % 2)


def quartic(pair, n, samples=2048):
	"""
	<S^4>, the first quantity that varies BETWEEN solutions.

	<S> = 0 identically (the shift removed the DC term, so there is no area).
	<S^2> = N exactly for every solution (no harmonic repeats), so the energy
	only separates solutions from non-solutions -- never one solution from
	another. <S^4> does vary, and it is exactly

		<S^4> = (3E + 4T) / 8            verified to machine precision

		E = #{(k1,k2,k3,k4) : k1 + k2 = k3 + k4}    additive energy
		T = #{(k1,k2,k3,k4) : k1 + k2 + k3 = k4}

	over the occupied harmonics. Solutions keep E bounded ABOVE -- they are more
	Sidon-like than the moments require. The lower end coincides, so this is a
	ceiling, not a two-sided constraint:

		N=8   soluciones [745.5, 780.5]   factibles [745.5, 806.5]
		N=10  soluciones [1443.5, 1520.5] factibles [1443.5, 1526.0]
	"""
	x = 2 * np.pi * np.arange(samples) / samples
	return float((pair_wave(pair, n, x) ** 4).mean())


def report(n=7):
	m0, m1, m2 = moments(n)
	x = np.linspace(0, np.pi, 1200)

	print(f'N = {n}')
	print(f'  M0 = {m0}   M1 = {m1}   M2 = {m2}      (invariantes)')

	pairs = feasible_pairs(n)
	bound_waves = np.array([pair_wave(p, n, x) for p in pairs])
	lo_b, hi_b = envelope(bound_waves)

	sol, other = all_perm_waves(n, x)
	lo_s, hi_s = envelope(sol)
	lo_a, hi_a = envelope(np.vstack([sol, other]))

	print(f'  pares factibles por momentos : {len(pairs)}')
	print(f'  soluciones reales            : {len(sol)}')
	assert (lo_b <= lo_s + 1e-9).all() and (hi_b >= hi_s - 1e-9).all(), 'cota violada'
	print('  la cota contiene a todas las soluciones: OK')

	tail = x > 0.5
	print(f'\n  para x > 0.5:')
	print(f'    |S| max sobre soluciones      = {np.abs(sol[:, tail]).max():.3f}')
	print(f'    |S| max de la cota (momentos) = {max(np.abs(lo_b[tail]).max(), np.abs(hi_b[tail]).max()):.3f}')
	print(f'    |S| max sobre TODAS las perms = {np.abs(np.vstack([sol, other])[:, tail]).max():.3f}')
	print(f'    cota trivial                  = {2*n}')

	gap = np.abs(hi_b - hi_s)[tail].max()
	print(f'    holgura maxima cota-vs-real   = {gap:.3f}')
	return x, (lo_s, hi_s), (lo_b, hi_b), (lo_a, hi_a)


if __name__ == '__main__':
	report(7)
