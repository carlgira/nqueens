"""
Un representante por orbita de simetria, elegido para quedar BIEN REPARTIDO.

El convenio habitual (el menor lexicografico de cada orbita) amontona los
representantes en la zona baja del espacio. Si en vez de eso se elige, dentro de
cada orbita, el elemento mas cercano a una posicion objetivo equiespaciada, el
conjunto queda casi perfectamente uniforme -- y sigue siendo un representante
valido por orbita.
"""

import numpy as np
from math import factorial


def lex_index(p):
	n = len(p); idx = 0
	for i in range(n):
		c = sum(1 for j in range(i + 1, n) if p[j] < p[i])
		idx += c * factorial(n - 1 - i)
	return idx


def sols(n):
	out, cur, full = [], [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if r == n: out.append(tuple(cur)); return
		f = ~(cols | dp | dm) & full
		while f:
			b = f & -f; f ^= b; c = b.bit_length() - 1
			cur.append(c); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	return out


def group(p):
	"""Las 8 imagenes de p bajo el grupo diedrico del tablero."""
	n = len(p); out = set(); q = tuple(p)
	for _ in range(4):
		out.add(q)
		out.add(tuple(n - 1 - c for c in q))				# espejo de columnas
		inv = [0] * n
		for r, c in enumerate(q): inv[c] = r					# transpuesta
		q = tuple(inv)
		q = tuple(n - 1 - c for c in q)
	# cerrar el grupo componiendo hasta saturar
	changed = True
	while changed:
		changed = False
		for r_ in list(out):
			for img in (tuple(n - 1 - c for c in r_),
						tuple(r_[n - 1 - i] for i in range(n))):
				if img not in out: out.add(img); changed = True
			inv = [0] * n
			for r, c in enumerate(r_): inv[c] = r
			if tuple(inv) not in out: out.add(tuple(inv)); changed = True
	return out


def orbits(n):
	S = set(sols(n)); seen = set(); out = []
	for p in sorted(S):
		if p in seen: continue
		o = group(p) & S
		seen |= o
		out.append(sorted(o, key=lex_index))
	return out


def spread(idx, total):
	i = np.array(sorted(idx)); k = len(i)
	g = np.diff(np.concatenate([[0], i, [total]])) / (total / (k + 1))
	return dict(sd=g.std(), maxgap=g.max(), tiny=(g < .1).mean())


def choose_even(orbs, total):
	"""Un representante por orbita, el mas cercano a su posicion ideal."""
	K = len(orbs)
	targets = [(k + .5) * total / K for k in range(K)]
	info = sorted(((np.mean([lex_index(p) for p in o]), o) for o in orbs))
	return [min(o, key=lambda p: abs(lex_index(p) - t))
			for t, (_m, o) in zip(targets, info)]


if __name__ == '__main__':
	for n in (9, 10, 11, 12):
		O = orbits(n); T = factorial(n)
		tot = sum(len(o) for o in O)
		lexmin = [o[0] for o in O]
		even = choose_even(O, T)
		a = spread([lex_index(p) for p in lexmin], T)
		b = spread([lex_index(p) for p in even], T)
		sizes = sorted({len(o) for o in O})
		print(f'N={n}: {tot} soluciones · {len(O)} órbitas (tamaños {sizes})')
		print(f'   {"":>22} {"sd huecos":>10} {"hueco max":>10} {"% diminutos":>12}')
		print(f'   {"menor lexicográfico":>22} {a["sd"]:>10.2f} {a["maxgap"]:>10.2f} {a["tiny"]:>11.0%}')
		print(f'   {"elegido repartido":>22} {b["sd"]:>10.2f} {b["maxgap"]:>10.2f} {b["tiny"]:>11.0%}')
		print()
