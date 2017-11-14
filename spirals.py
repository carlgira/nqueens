import numpy as np
import nqueens
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec



# Linear combination of solutions between angles and solutions
def calculate_sum(n):
	monitor = nqueens.n_queens(n)
	phi = (1 + math.sqrt(5))/2
	c = pow(phi, 2/math.pi)
	angle = math.pi*2/n

	a = [angle*(i+1) for i in range(n)]
	r = [pow(c, -2*math.pi + l) for l in a]

	points = []
	for sol in monitor.all_solutions:
		s = [f*a[g] for f,g in zip(r, sol)]
		points.append(sum(s))

	points = [pp/max(points) for pp in points]
	points.sort()

	return points

#for nn in range(5,12):
#  p = calculate_sum(nn)
#  plt.plot(p)

#plt.show()

# Sum of distances (thetas) between points
def calculate_sum_theta_distance(n):
	monitor = nqueens.n_queens(n, True)
	angle = math.pi*2/n

	a = [angle*(i+1) for i in range(n)]

	#points = []
	#for sol in monitor.all_solutions:
	sol = monitor.all_solutions[0]
	thetas = [2*math.pi*(i+1) + (sol[i])*angle for i,j in zip(range(n), a)]

	return sum(thetas)


#monitor = nqueens.n_queens(14, True)
v = [106.81415022205297, 147.6548547187203, 194.7787445225672, 248.18581963359367, 307.8760800517997,
	 373.84952577718536, 446.1061568097506, 524.6459731494955, 609.4689747964198, 700.5751617505238,
	 797.9645340118074]


def create_polar_graphs(n):
	monitor = nqueens.n_queens(n)
	groups = monitor.group_solutions()
	print(len(groups))

	phi = (1 + math.sqrt(5))/2
	c = pow(phi, 2/math.pi)
	angle = math.pi*2/n

	fractions = [2/n*(i+1) for i in range(n)]
	a = [angle*(i+1) for i in range(n)]

	spiral_func = lambda t, l: math.pow(c, t - 12*math.pi + l)

	r = []
	r.append(spiral_func(8*math.pi + angle, 2*math.pi))
	for aa in a[:-1]:
		r.append(spiral_func(10*math.pi + angle, aa))

	for h in range(0,len(groups),2):
		hend = h + 2
		if h+2 > len(groups):
			hend = len(groups)

		print(h, hend)

		fig = plt.figure(figsize=(20, 20))

		for group,i in zip(groups[h:hend], range(len(groups[h:hend]))):

			for sol,j in zip(group, range(len(group))):
				ax = plt.subplot2grid((2, 8), (i, j), projection='polar')
				ax.set_title( str(np.array(sol)+1), va='bottom', fontdict = {'fontsize': 10})

				for aa, f in zip(a, fractions):
					theta = np.pi * np.arange(0, 12 - f, 0.01)
					r_values = [spiral_func(t, aa) for t in theta]
					ax.plot(theta, r_values, c='b')

				theta = np.pi * np.arange(0, 2*math.pi, 0.01)
				for rr in r:
					ax.plot(theta, [rr]*len(theta),  c='b')

				tt = [a[v] for v in sol]
				print(sol)
				print(tt, r)
				ax.scatter(tt, r, s=70, c='r')

				ax.set_yticklabels([])
				#ax.grid(False)
				ax.set_xticklabels([])
				ax.set_yticklabels([])

				ax.set_rmax(1)

		plt.tight_layout()
		#plt.show()
		#plt.savefig('foo' + str(h) + '.png')

create_polar_graphs(5)
