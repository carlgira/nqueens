import numpy as np
import nqueens
import math

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

	# [2*math.pi*(i+1) + (sol[i])*angle for i,j in zip(range(n), a)]

	return sum(thetas)


#monitor = nqueens.n_queens(14, True)
v = [106.81415022205297, 147.6548547187203, 194.7787445225672, 248.18581963359367, 307.8760800517997,
	 373.84952577718536, 446.1061568097506, 524.6459731494955, 609.4689747964198, 700.5751617505238,
	 797.9645340118074]

def PolyArea(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def create_polar_graphs(n):
	monitor = nqueens.n_queens(n)
	groups = monitor.group_solutions()
	#print(np.shape(groups))
	#groups[0].append( [3, 0, 4, 1, 2, 5] )
	#print(np.shape(groups))


	phi = (1 + math.sqrt(5))/2
	c = pow(phi, 2/math.pi)
	angle = math.pi*2/n

	fractions = [2/n*(i+1) for i in range(n)]
	a = [angle*(i+1) for i in range(n)]

	spiral_func = lambda t, l: math.pow(c, t - 12*math.pi + l)

	c1 = pow(phi, math.pi/(3*n))
	spiral_func1 = lambda t, l: math.pow(c1, t - 12*math.pi + l)

	r = []
	areas = []
	r.append(spiral_func(8*math.pi + angle, 2*math.pi))
	for aa in a[:-1]:
		r.append(spiral_func(10*math.pi + angle, aa))

	for h in range(0,len(groups),2):
		hend = h + 2
		if h+2 > len(groups):
			hend = len(groups)


		for group,i in zip(groups[h:hend], range(len(groups[h:hend]))):

			for sol,j in zip(group, range(len(group))):
				tt = [a[v] for v in sol]

				if sol == [2,0,3,1,4]:
					f = spiral_func1(10*math.pi + angle, 2*math.pi)





					#x = [r[sol.index(d)]*math.cos(tt[sol.index(d)]) for d in range(len(sol))]
					#y = [r[sol.index(d)]*math.sin(tt[sol.index(d)]) for d in range(len(sol))]
					#area = PolyArea(x,y)
					#print("a", area ,sol, x, y)
					#areas.append(area)








	print("len-areas",  len(set(areas)), set(areas))

create_polar_graphs(5)
