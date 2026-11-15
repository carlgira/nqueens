"""
Conteo de soluciones por SIMETRIA: el tamano de la orbita bajo D4.

D4 (las 8 isometrias del tablero) actua sobre las soluciones. Una orbita tiene
tamano 8/|estabilizador|, y el estabilizador solo puede ser C1, C2 o C4 —
**ninguna solucion tiene simetria de reflexion**. Razon, en una linea: si el
tablero es simetrico respecto de la diagonal principal, la permutacion es una
involucion, y una casilla (i,j) con i!=j obliga a (j,i), que esta en la MISMA
antidiagonal i+j. Conflicto. Igual para las otras tres reflexiones.

Asi que solo hay tres clases:

	orbita 8  estabilizador trivial          "ordinarias"
	orbita 4  simetricas a 180 grados
	orbita 2  simetricas a 90 grados         (solo si N = 0 o 1 mod 4)

Contando SOLUCIONES (no orbitas), con S2 = fijadas por rot180 y S4 = por rot90:

	a2 = S4/2        a4 = (S2-S4)/4        a8 = (Q-S2)/8
	fundamentales q(N) = a8 + a4 + a8 ... = a8 + a4 + a2

Las dos condiciones de simetria, en forma de permutacion (fila i -> columna s[i],
base 0):

	rot180:  s[n-1-i] = n-1-s[i]
	rot90 :  s[s[i]]  = n-1-i          es decir  s o s = reverso

De la segunda sale sola la condicion conocida: s tiene orden 4, sus ciclos son
4-ciclos (mas un punto fijo central si N es impar), luego N = 0 o 1 mod 4.
"""

import sys

QN = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596,
      2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884,
      314666222712, 2691008701644, 24233937684440, 227514171973736,
      2207893435808352, 22317699616364044, 234907967154122528]


# ------------------------------------------------- referencia: fuerza bruta

def all_solutions(n):
	"""Todas las soluciones, backtracking con mascaras de bits."""
	out = []
	s = [0] * n
	full = (1 << n) - 1

	def go(i, cols, d1, d2):
		if i == n:
			out.append(tuple(s))
			return
		free = full & ~(cols | d1 | d2)
		while free:
			b = free & -free
			free ^= b
			s[i] = b.bit_length() - 1
			go(i + 1, cols | b, ((d1 | b) << 1) & full, (d2 | b) >> 1)
	go(0, 0, 0, 0)
	return out


def classify_bruteforce(n):
	"""(Q, S2, S4) recorriendo todas las soluciones."""
	sols = all_solutions(n)
	S2 = S4 = 0
	for s in sols:
		if all(s[n - 1 - i] == n - 1 - s[i] for i in range(n)):
			S2 += 1
			if all(s[s[i]] == n - 1 - i for i in range(n)):
				S4 += 1
	return len(sols), S2, S4


# ------------------------------------- busqueda directa de las simetricas

def count_rot180(n):
	"""Soluciones con s[n-1-i] = n-1-s[i]. Coloca las filas por parejas
	(i, n-1-i): elegida la columna de una, la otra queda forzada."""
	full = (1 << n) - 1
	pairs = [(i, n - 1 - i) for i in range(n // 2)]
	cnt = 0

	def place(cells, k, cols, dp, dm):
		"""cells: lista de (fila, col) a colocar de golpe; comprueba y sigue."""
		nonlocal cnt
		c2 = cols
		p2, m2 = dp, dm
		for (r, c) in cells:
			b = 1 << c
			bp = 1 << (r + c)
			bm = 1 << (r - c + n - 1)
			if (c2 & b) or (p2 & bp) or (m2 & bm):
				return
			c2 |= b
			p2 |= bp
			m2 |= bm
		go(k, c2, p2, m2)

	def go(k, cols, dp, dm):
		nonlocal cnt
		if k == len(pairs):
			cnt += 1
			return
		i, j = pairs[k]
		free = full & ~cols
		f = free
		while f:
			b = f & -f
			f ^= b
			c = b.bit_length() - 1
			place([(i, c), (j, n - 1 - c)], k + 1, cols, dp, dm)

	if n % 2 == 0:
		go(0, 0, 0, 0)
	else:
		m = (n - 1) // 2                      # fila central forzada al centro
		place([(m, m)], 0, 0, 0, 0)
	return cnt


def count_rot90(n):
	"""Soluciones con s[s[i]] = n-1-i. Coloca orbitas completas de 4 casillas
	bajo (i,j) -> (j, n-1-i)."""
	if n % 4 not in (0, 1):
		return 0
	full = (1 << n) - 1
	cnt = 0

	def go(rows, cols, dp, dm):
		nonlocal cnt
		if rows == full:
			cnt += 1
			return
		i = (~rows & full)
		i = (i & -i).bit_length() - 1         # menor fila libre
		f = full & ~cols
		while f:
			b = f & -f
			f ^= b
			j = b.bit_length() - 1
			orb = [(i, j), (j, n - 1 - i), (n - 1 - i, n - 1 - j), (n - 1 - j, i)]
			r2, c2, p2, m2 = rows, cols, dp, dm
			ok = True
			for (r, c) in orb:
				br, bc = 1 << r, 1 << c
				bp, bm = 1 << (r + c), 1 << (r - c + n - 1)
				if (r2 & br) or (c2 & bc) or (p2 & bp) or (m2 & bm):
					ok = False
					break
				r2 |= br
				c2 |= bc
				p2 |= bp
				m2 |= bm
			if ok:
				go(r2, c2, p2, m2)

	if n % 4 == 0:
		go(0, 0, 0, 0)
	else:
		m = (n - 1) // 2                      # el centro es punto fijo de rot90
		go(1 << m, 1 << m, 1 << (2 * m), 1 << (n - 1))
	return cnt


def table(nmax, check_upto=12):
	print("control: busqueda directa vs fuerza bruta")
	for n in range(4, check_upto + 1):
		Q, S2, S4 = classify_bruteforce(n)
		d2, d4 = count_rot180(n), count_rot90(n)
		tag = "ok" if (d2, d4) == (S2, S4) else f"DISCREPA ({d2},{d4})"
		print(f"  N={n:2d}  Q={Q:8d}  S2={S2:5d}  S4={S4:4d}   {tag}")

	print("\n(N=1 es el unico caso con reflexion: tablero 1x1, orbita de tamano 1.)")
	print("\n N        Q(N)          orbita 8      orbita 4   orbita 2   fundamentales")
	for n in range(4, nmax + 1):
		Q = QN[n - 1]
		S2, S4 = count_rot180(n), count_rot90(n)
		a2, a4, a8 = S4 // 2, (S2 - S4) // 4, (Q - S2) // 8
		assert 8 * a8 + 4 * a4 + 2 * a2 == Q, n
		print(f"{n:2d} {Q:20d} {a8:14d} {a4:9d} {a2:9d} {a8+a4+a2:15d}")


if __name__ == "__main__":
	table(int(sys.argv[1]) if len(sys.argv) > 1 else 18)
