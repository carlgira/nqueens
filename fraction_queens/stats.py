"""Que numeros salen: estadistica de x(sigma) para soluciones vs permutaciones."""
import sys, os, math, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import frac

NS = list(range(5, 13))
SAMPLE = 200000


def xs(perms, n):
	return np.array([frac.frac(p, n) for p in perms])


def ks(a, b):
	"""Distancia Kolmogorov-Smirnov entre dos muestras."""
	a = np.sort(a); b = np.sort(b)
	allv = np.concatenate([a, b])
	ca = np.searchsorted(a, allv, 'right') / len(a)
	cb = np.searchsorted(b, allv, 'right') / len(b)
	return float(np.max(np.abs(ca - cb)))


def report():
	print(f"{'N':>3} {'Q(N)':>7} {'media_sol':>10} {'media_per':>10} "
	      f"{'sd_sol':>8} {'sd_per':>8} {'KS':>6} {'gap_max_sol':>12} {'gap_esp':>10}")
	rows = {}
	for n in NS:
		sols = frac.solutions(n)
		perms = (frac.permutations_all(n) if n <= 8
		         else frac.permutations_sample(n, SAMPLE))
		xs_s, xs_p = xs(sols, n), xs(perms, n)
		g = np.diff(np.sort(xs_s))
		print(f"{n:>3} {len(sols):>7} {xs_s.mean():>10.6f} {xs_p.mean():>10.6f} "
		      f"{xs_s.std():>8.5f} {xs_p.std():>8.5f} {ks(xs_s, xs_p):>6.3f} "
		      f"{g.max():>12.5f} {1/len(sols):>10.5f}")
		rows[n] = (sols, perms, xs_s, xs_p)
	return rows


def digits(rows):
	print("\n--- distribucion del PRIMER digito (banda de x), en % ---")
	for n in NS:
		sols, perms, _, _ = rows[n]
		hs = np.bincount([s[0] for s in sols], minlength=n) / len(sols) * 100
		print(f"N={n:>2} sol  " + " ".join(f"{v:5.1f}" for v in hs)
		      + f"   uniforme={100/n:.1f}")


def dims(rows):
	print("\n--- dimensiones del conjunto de fracciones (cajas de lado N^-d) ---")
	print(f"{'N':>3} {'d':>3} {'W_d':>8} {'dim_caja':>9} {'dim_info':>9}")
	for n in NS:
		sols = rows[n][0]
		for d in range(1, n + 1):
			pref = {}
			for s in sols:
				pref[s[:d]] = pref.get(s[:d], 0) + 1
			W = len(pref)                         # cajas realmente ocupadas
			p = np.array(list(pref.values()), float)
			p /= p.sum()
			H = -(p * np.log(p)).sum()
			if d in (1, n // 2, n):
				print(f"{n:>3} {d:>3} {W:>8} {math.log(W)/(d*math.log(n)):>9.4f} "
				      f"{H/(d*math.log(n)):>9.4f}")
		print()


def rationals(rows):
	print("--- el racional p/q: cuanto se simplifica y como es su fraccion continua ---")
	print(f"{'N':>3} {'q<N^N %':>8} {'len_cf_sol':>11} {'len_cf_per':>11} "
	      f"{'maxA_sol':>9} {'maxA_per':>9} {'geoA_sol':>9} {'geoA_per':>9}")
	for n in NS:
		sols, perms, _, _ = rows[n]
		k = min(len(sols), 2000)
		pe = perms[:k]
		so = sols[:k]
		def cf_stats(ps):
			L, mx, lg = [], [], []
			for s in ps:
				p, q = frac.reduced(s, n)
				a = frac.cont_frac(p, q)[1:]     # sin la parte entera (=0)
				if not a:
					continue
				L.append(len(a)); mx.append(max(a))
				lg.append(sum(math.log(v) for v in a) / len(a))
			return np.mean(L), np.mean(mx), math.exp(np.mean(lg))
		red = sum(1 for s in sols if frac.reduced(s, n)[1] != n ** n) / len(sols) * 100
		a1 = cf_stats(so); a2 = cf_stats(pe)
		print(f"{n:>3} {red:>8.1f} {a1[0]:>11.2f} {a2[0]:>11.2f} "
		      f"{a1[1]:>9.1f} {a2[1]:>9.1f} {a1[2]:>9.3f} {a2[2]:>9.3f}")




def sturmian_signature(s, n):
	"""Pasos sigma(k+1)-sigma(k) mod N: 1 valor = paso constante, 2 = Sturmiana."""
	steps = {(s[k + 1] - s[k]) % n for k in range(n - 1)}
	return len(steps)


def near_rationals(rows):
	"""Que soluciones estan anormalmente cerca de un racional simple?"""
	print("\n--- max cociente parcial de la fraccion continua vs estructura ---")
	print(f"{'N':>3} {'#pasos':>7} {'cuantas':>8} {'maxA medio':>11} {'maxA mediana':>13}")
	for n in NS:
		sols = rows[n][0]
		by = {}
		for s in sols:
			p, q = frac.reduced(s, n)
			a = frac.cont_frac(p, q)[1:]
			by.setdefault(sturmian_signature(s, n), []).append(max(a) if a else 0)
		for k in sorted(by):
			v = np.array(by[k], float)
			print(f"{n:>3} {k:>7} {len(v):>8} {v.mean():>11.1f} {np.median(v):>13.1f}")
		print()


if __name__ == '__main__':
	rows = report()
	digits(rows)
	dims(rows)
	rationals(rows)
	near_rationals(rows)
