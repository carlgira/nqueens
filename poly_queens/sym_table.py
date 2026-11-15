"""Tabla final: soluciones por tamano de orbita bajo D4, con cache."""
import json, os
from .symmetry import QN, count_rot180, count_rot90

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "sym.json")


def data(nmax):
	d = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
	for n in range(4, nmax + 1):
		if str(n) not in d:
			d[str(n)] = [count_rot180(n), count_rot90(n)]
			json.dump(d, open(CACHE, "w"))
			print(f"  N={n} calculado", flush=True)
	return {int(k): v for k, v in d.items()}


def show(nmax=27):
	d = data(nmax)
	print(f"{'N':>3} {'Q(N)':>21} {'orbita 8':>14} {'orbita 4':>9} {'orbita 2':>8}"
	      f" {'fundament.':>14}")
	for n in sorted(d):
		if n > len(QN):
			continue
		S2, S4 = d[n]
		Q = QN[n - 1]
		a2, a4, a8 = S4 // 2, (S2 - S4) // 4, (Q - S2) // 8
		assert 8 * a8 + 4 * a4 + 2 * a2 == Q, n
		print(f"{n:3d} {Q:21d} {a8:14d} {a4:9d} {a2:8d} {a8+a4+a2:14d}")


if __name__ == "__main__":
	import sys
	show(int(sys.argv[1]) if len(sys.argv) > 1 else 27)
