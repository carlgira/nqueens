"""
The two-variable wave: the fix for the decodability defect.

The 1D wave S(x) is a PROJECTION of the board along a direction. Projections
lose the pairing -- which d+ diagonal crosses which d- diagonal -- so different
boards collapse onto the same S (92 solutions -> 58 waves at N=8).

Adding row and column waves does not help: every permutation matrix has the
exact same row and column projections (all ones), so they carry zero bits.

The fix is not another projection but one dimension up:

	W(u, v) = sum_i exp( 2j*pi*(i*u + perm[i]*v) / M ),      M = 2N-1

This is the 2D DFT of the board matrix, so it is invertible -- the board comes
back with a plain inverse transform. And the two diagonal witnesses are simply
its restrictions to the two diagonal LINES of the (u,v) plane:

	<|W(t,  t)|^2> = N + 2*conflicts(d+)        u =  v
	<|W(t, -t)|^2> = N + 2*conflicts(d-)        u = -v

So the old 1D wave was this object cut along a line. Cutting is where the
information went.

Verified at N=7: 5040 distinct W for 5040 permutations, zero collisions.
"""

import numpy as np


def wave2d(perm, m=None):
	"""W(u,v) over the (2N-1)x(2N-1) grid. Complex, invertible."""
	n = len(perm)
	m = m or 2 * n - 1
	board = np.zeros((m, m))
	board[np.arange(n), np.asarray(perm)] = 1
	return np.fft.ifft2(board) * m ** 2


def decode2d(w, n):
	"""Recover the board, then the permutation. Exact."""
	m = w.shape[0]
	board = np.rint(np.real(np.fft.fft2(w)) / m ** 2).astype(int)[:n, :n]
	return [int(np.argmax(board[r])) for r in range(n)]


def witnesses(w, n):
	"""(conflicts on d+, conflicts on d-), read off the two diagonal cuts."""
	m = w.shape[0]
	t = np.arange(m)
	up = (np.abs(w[t, t]) ** 2).mean()
	dn = (np.abs(w[t, (-t) % m]) ** 2).mean()
	return (up - n) / 2, (dn - n) / 2


def total_conflicts(w, n):
	a, b = witnesses(w, n)
	return a + b


if __name__ == '__main__':
	from itertools import permutations
	from diag_waves.diag_waves import channels, conflicts

	n = 8
	p, q = [0, 4, 7, 5, 2, 6, 1, 3], [0, 6, 3, 5, 7, 1, 4, 2]
	print(f'los dos tableros que compartian la onda 1D:')
	for name, perm in (('p', p), ('q', q)):
		w = wave2d(perm)
		dp, dm = channels(perm)
		print(f'  {name}: decodifica a {decode2d(w, n)}  '
			  f'testigos {tuple(round(v) for v in witnesses(w, n))}  '
			  f'real {(conflicts(dp), conflicts(dm))}')
	print(f'  misma onda 2D? {np.allclose(wave2d(p), wave2d(q))}')

	seen = {np.round(wave2d(pp), 6).tobytes() for pp in permutations(range(7))}
	print(f'\nN=7: {len(seen)} ondas 2D distintas para 5040 permutaciones')
