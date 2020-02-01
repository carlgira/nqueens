import numpy as np
import nqueens
import math
import random

phi = (1 + math.sqrt(5))/2
c = pow(phi, 2/math.pi)
spiral1 = lambda t, l: math.pow(c, t - 12*math.pi + l)
spiral2 = lambda t, l: math.pow(c, -(t - 2*math.pi + l))

def gen_c():
	while True:
		n = random.randint(1, 10)
		d = random.randint(1, 10)

		if n/d > 0.5 and n/d < 2:
			return n, d


def eval_entity(sol, points):
	sol = {"n": 1, "d": 2, "spiral": True, "spiral_param": 2}






def find_spiral(n):
	monitor = nqueens.n_queens(n)



	angle = math.pi*2/n

	fractions = [2/n*(i+1) for i in range(n)]
	a = [angle*(i+1) for i in range(n)]

	r = []
	r.append(spiral1(8*math.pi + angle, 2*math.pi))
	for aa in a[:-1]:
		r.append(spiral1(10*math.pi + angle, aa))



	points = []





