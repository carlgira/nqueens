"""
Poda exacta por ESPECTRO PENDIENTE.

A las N-r reinas que faltan les hacen falta N-r diagonales d+ distintas, N-r
diagonales d- distintas y N-r columnas distintas. Si de alguna de las tres hay
menos disponibles, el estado esta muerto -- con certeza, no heuristicamente.

Se compara contra:
  - DFS normal
  - forward-check clasico (alguna fila sin opciones)
"""

import numpy as np


def count(n, mode='plain'):
	"""Cuenta TODAS las soluciones. mode: plain | fc | spec"""
	full = (1 << n) - 1
	nodes = [0]; found = [0]

	def rec(r, cols, dp, dm):
		nodes[0] += 1
		if r == n:
			found[0] += 1; return
		if mode != 'plain' and r < n:
			falta = n - r
			seenA = 0; seenB = 0; seenC = 0; dead = False
			for i in range(r, n):
				free = ~(cols | (dp << (i - r)) | (dm >> (i - r))) & full
				if free == 0:
					dead = True; break
				if mode == 'spec':
					seenC |= free
					f = free
					while f:
						b = f & -f; f ^= b; c = b.bit_length() - 1
						seenA |= 1 << (i + c)
						seenB |= 1 << (i - c + n)
			if dead: return
			if mode == 'spec':
				if (bin(seenA).count('1') < falta or bin(seenB).count('1') < falta
						or bin(seenC).count('1') < falta):
					return
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b
			rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1)

	rec(0, 0, 0, 0)
	return nodes[0], found[0]


if __name__ == '__main__':
	import time
	print(f'{"N":>3} {"sols":>9} | ' + ' '.join(
		f'{h:>24}' for h in ('DFS normal', 'forward-check', 'espectro pendiente')))
	print(f'{"":>3} {"":>9} | ' + ' '.join(f'{"nodos":>12}{"seg":>12}' for _ in range(3)))
	print('-' * 92)
	for n in range(8, 14):
		row = []
		for mode in ('plain', 'fc', 'spec'):
			t0 = time.time(); nd, f = count(n, mode); s = time.time() - t0
			row.append((nd, s, f))
		assert row[0][2] == row[1][2] == row[2][2], 'perdio soluciones!'
		print(f'{n:>3} {row[0][2]:>9,} | ' + ' '.join(
			f'{nd:>12,}{s:>12.2f}' for nd, s, _ in row))


def match_count(A, b, n, mode='plain'):
	"""Empareja un espectro (A,b) con un tablero. Devuelve (nodos, existe)."""
	SA = 0
	for v in A: SA |= 1 << v
	SB = 0
	for v in b: SB |= 1 << v
	full = (1 << n) - 1
	nodes = [0]
	usedA = [0]; usedB = [0]

	def rec(r, cols):
		nodes[0] += 1
		if r == n: return True
		if mode == 'spec':
			falta = n - r
			seenA = seenB = seenC = 0
			for i in range(r, n):
				got = 0
				for c in range(n):
					if cols >> c & 1: continue
					u = i + c + 1; v = i - c + n
					if not (SA >> u & 1) or not (SB >> v & 1): continue
					if usedA[0] >> u & 1 or usedB[0] >> v & 1: continue
					got |= 1 << c; seenA |= 1 << u; seenB |= 1 << v
				if got == 0: return False
				seenC |= got
			if (bin(seenA).count('1') < falta or bin(seenB).count('1') < falta
					or bin(seenC).count('1') < falta):
				return False
		for c in range(n):
			if cols >> c & 1: continue
			u = i_u = r + c + 1; v = r - c + n
			if not (SA >> u & 1) or not (SB >> v & 1): continue
			if usedA[0] >> u & 1 or usedB[0] >> v & 1: continue
			usedA[0] |= 1 << u; usedB[0] |= 1 << v
			if rec(r + 1, cols | 1 << c): return True
			usedA[0] &= ~(1 << u); usedB[0] &= ~(1 << v)
		return False

	ok = rec(0, 0)
	return nodes[0], ok
