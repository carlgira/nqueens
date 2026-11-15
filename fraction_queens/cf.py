"""
Medidas sobre la fraccion continua de x(sigma) = M/N^N.

Para un racional p/q "al azar" la teoria dice:
  - longitud de la FC ~ (12 ln2/pi^2) ln q = 0.8427 ln q   (Heilbronn)
  - media geometrica de los cocientes parciales -> 2.6854   (Khinchin)
  - ningun cociente domina: ln(max a_i)/ln q ~ 0

Definimos por permutacion:
  L  = len(FC) / (0.8427 ln q)     ~1 si es generico
  C  = ln(max a_i) / ln q          fraccion de la informacion que se lleva
                                   el mayor cociente parcial; ~0 si generico
  G  = media geometrica de a_i     ~2.685 si generico
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import frac

HEILBRONN = 12 * math.log(2) / math.pi ** 2      # 0.84271


def measures(sigma, n):
	p, q = frac.reduced(sigma, n)
	a = frac.cont_frac(p, q)[1:]
	if not a:
		return None
	lq = math.log(q)
	L = len(a) / (HEILBRONN * lq)
	C = math.log(max(a)) / lq
	G = math.exp(sum(math.log(v) for v in a) / len(a))
	return L, C, G


def table(ns=range(6, 13), sample=4000):
	print(f"{'N':>3} {'Q(N)':>7} | {'L_sol':>6} {'L_per':>6} | {'C_sol':>6} "
	      f"{'C_per':>6} {'C_max_sol':>10} {'C_max_per':>10} | {'G_sol':>6} {'G_per':>6}"
	      f" | {'%C>0.5 sol':>10} {'%C>0.5 per':>10}")
	out = {}
	for n in ns:
		sols = frac.solutions(n)
		perms = (frac.permutations_all(n) if n <= 8
		         else frac.permutations_sample(n, max(sample, len(sols)), seed=1))
		S = np.array([measures(s, n) for s in sols], float)
		P = np.array([measures(s, n) for s in perms], float)
		out[n] = (sols, S, P)
		print(f"{n:>3} {len(sols):>7} | {S[:,0].mean():>6.3f} {P[:,0].mean():>6.3f} | "
		      f"{S[:,1].mean():>6.3f} {P[:,1].mean():>6.3f} {S[:,1].max():>10.3f} "
		      f"{P[:,1].max():>10.3f} | {S[:,2].mean():>6.3f} {P[:,2].mean():>6.3f} | "
		      f"{100*(S[:,1]>.5).mean():>10.2f} {100*(P[:,1]>.5).mean():>10.2f}")
	return out


if __name__ == '__main__':
	out = table()
	print("\n--- las mas 'algebraicas' (C alto = casi un racional simple) ---")
	for n in (10, 11, 12):
		sols, S, _ = out[n]
		idx = np.argsort(-S[:, 1])[:3]
		for i in idx:
			s = sols[i]
			st = sorted({(s[k+1]-s[k]) % n for k in range(n-1)})
			print(f"N={n} C={S[i,1]:.3f} L={S[i,0]:.3f} pasos={st} {list(s)}")
		print()
