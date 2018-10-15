import numpy as np
import math
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance

def create_polar_graphs(n):

	phi = (1 + math.sqrt(5))/2
	c = pow(phi, 2/math.pi)
	#c1 = math.pow(phi, 2/(3*math.pi)) # 5 OK 1.107512291

	c1 = math.pow(phi, 1/(2*math.pi)) # 7 OK 1.0795963709
	#c1 = 1.095 # 6
	#c1 = 1.061242868 # 8


	angle = math.pi*2/n
	angles = [angle*(i+1) for i in range(n)]
	fractions = [2/n*(i+1) for i in range(n)]

	spiral_func_i = lambda t, c, l: math.pow(c, t - 12*math.pi + l)
	spiral_func_d = lambda t, c, l: math.pow(c, -(t - 2*math.pi + l))

	r = []
	r.append(spiral_func_i(8*math.pi + angle,c , 2*math.pi))
	for a in angles[:-1]:
		r.append(spiral_func_i(10*math.pi + angle, c, a))

	thetas = np.array(angles*n).reshape(n,n).T
	rs = np.array(r*n).reshape(n,n)

	xc = rs*np.cos(thetas)
	yc = rs*np.sin(thetas)

	sols = []

	fig = plt.figure()

	for theta, fraction, e in zip(angles, fractions, range(n)):

		sol_i = []
		for f,o in zip(fractions,range(n)):
			thetasc_i = np.pi * np.arange(f, 12, 2)
			rsc_i = [spiral_func_i(t, c1 ,theta) for t in thetasc_i]

			ax = plt.subplot2grid((1, 1), (0, 0), projection='polar')
			ax.scatter(thetas[o], rs[o], c='y')
			ax.scatter(thetasc_i, rsc_i, c='b')

			theta_i_g = np.pi * np.arange(f, 12 - fraction, 0.01)
			r_values_i_g = [spiral_func_i(t, c ,theta) for t in theta_i_g]
			rsc_i_g_l = [spiral_func_i(t, c1, theta) for t in theta_i_g]
			ax.plot(theta_i_g, r_values_i_g, c='b')
			ax.plot(theta_i_g, rsc_i_g_l, c='r')



			xc_i = (rsc_i*np.cos(thetasc_i))
			yc_i = (rsc_i*np.sin(thetasc_i))

			cords_i = [[x,y] for x,y in zip(xc_i, yc_i)]
			cords = [[x,y] for x,y in zip(xc[o], yc[o])]

			dist_i = distance.cdist(cords, cords_i)

			sol_i_pos = np.unravel_index(dist_i.argmin(), dist_i.shape)[0]
			sol_i.append(sol_i_pos)

			#plt.show()

		print("r", sol_i, is_valid(n, sol_i))
		if is_valid(n, sol_i) and sol_i not in sols:
			sols.append(sol_i)

		sol_d = []
		for f,o in zip(fractions,range(n)):
			thetasc_d = np.pi * np.arange(f, n+2, 2)
			rsc_d = np.array([spiral_func_d(t, c1, 2*math.pi - theta) for t in thetasc_d])


			thetasc_d = thetasc_d[np.where(rsc_d > r[0])]
			rsc_d = rsc_d[np.where(rsc_d > r[0])]



			ax = plt.subplot2grid((1, 1), (0, 0), projection='polar')

			ax.scatter(thetas[o], rs[o], c='y')
			ax.scatter(thetasc_d, rsc_d, c='g')

			theta_d_g = np.pi * np.arange(fraction, n+2, 0.01)
			r_values_d_g = [spiral_func_d(t, c , 2*math.pi - theta) for t in theta_d_g]
			rsc_d_g_l = [spiral_func_d(t, c1, 2*math.pi - theta) for t in theta_d_g]
			ax.plot(theta_d_g, r_values_d_g, c='b')

			ax.plot(theta_d_g, rsc_d_g_l, c='g')



			xc_d = (rsc_d*np.cos(thetasc_d))
			yc_d = (rsc_d*np.sin(thetasc_d))


			cords_d = [[x,y] for x,y in zip(xc_d, yc_d)]
			cords = [[x,y] for x,y in zip(xc[o], yc[o])]

			dist_d = distance.cdist(cords, cords_d)



			sol_d_pos = np.unravel_index(dist_d.argmin(), dist_d.shape)
			sol_d.append(sol_d_pos[0])


			#plt.show()


		print("g", sol_d, is_valid(n, sol_d))
		if is_valid(n, sol_d) and sol_d not in sols:
			sols.append(sol_d)


	return sols


def create_polar_graphs_c1(n, c1):

	phi = (1 + math.sqrt(5))/2
	c = pow(phi, 2/math.pi)

	angle = math.pi*2/n
	angles = [angle*(i+1) for i in range(n)]
	fractions = [2/n*(i+1) for i in range(n)]

	spiral_func_i = lambda t, c, l: math.pow(c, t - 12*math.pi + l)
	spiral_func_d = lambda t, c, l: math.pow(c, -(t - 2*math.pi + l))

	r = []
	r.append(spiral_func_i(8*math.pi + angle,c , 2*math.pi))
	for a in angles[:-1]:
		r.append(spiral_func_i(10*math.pi + angle, c, a))

	thetas = np.array(angles*n).reshape(n,n).T
	rs = np.array(r*n).reshape(n,n)

	xc = rs*np.cos(thetas)
	yc = rs*np.sin(thetas)

	sols = []

	for theta, fraction, e in zip(angles, fractions, range(n)):

		sol_i = []
		for f,o in zip(fractions,range(n)):
			thetasc_i = np.pi * np.arange(f, 12, 2)
			rsc_i = np.array([spiral_func_i(t, c1 ,theta) for t in thetasc_i])

			thetasc_i = thetasc_i[np.where(rsc_i > r[0])]
			rsc_i = rsc_i[np.where(rsc_i > r[0])]

			xc_i = (rsc_i*np.cos(thetasc_i))
			yc_i = (rsc_i*np.sin(thetasc_i))

			cords_i = [[x,y] for x,y in zip(xc_i, yc_i)]
			cords = [[x,y] for x,y in zip(xc[o], yc[o])]

			dist_i = distance.cdist(cords, cords_i)

			sol_i_pos = np.unravel_index(dist_i.argmin(), dist_i.shape)[0]
			sol_i.append(sol_i_pos)

		if is_valid(n, sol_i) and sol_i not in sols:
			sols.append(sol_i)

		sol_d = []
		for f,o in zip(fractions,range(n)):
			thetasc_d = np.pi * np.arange(f, n+2, 2)
			rsc_d = np.array([spiral_func_d(t, c1, 2*math.pi - theta) for t in thetasc_d])

			thetasc_d = thetasc_d[np.where(rsc_d > r[0])]
			rsc_d = rsc_d[np.where(rsc_d > r[0])]

			xc_d = (rsc_d*np.cos(thetasc_d))
			yc_d = (rsc_d*np.sin(thetasc_d))


			cords_d = [[x,y] for x,y in zip(xc_d, yc_d)]
			cords = [[x,y] for x,y in zip(xc[o], yc[o])]

			dist_d = distance.cdist(cords, cords_d)

			sol_d_pos = np.unravel_index(dist_d.argmin(), dist_d.shape)
			sol_d.append(sol_d_pos[0])

		if is_valid(n, sol_d) and sol_d not in sols:
			sols.append(sol_d)

	return sols


def is_valid(n, vec):
	cols = range(n)
	return n == len(set(vec)) and (n == len(set(vec[i]+i for i in cols)) == len(set(vec[i]-i for i in cols)))


def check_c1(n):

	#c1 = 1.095 # 6

	c1s = np.arange(1.02, 1.08, 0.000001)  # 8

	for c1 in c1s:
		sols = create_polar_graphs_c1(n, c1)

		if len(sols) > 0:
			print("c1", n, c1, len(sols))


check_c1(8)


#def min_fun(vv):
#c1 = math.pow(phi, vv*math.pi) # 9 1.0523
#thetasc = [math.log(rc, c1) + 12*math.pi - theta for rc in rsc]

#xc1 = rsc*np.cos(thetasc)
#yc1 = rsc*np.sin(thetasc)

#dist = np.sqrt((xc-xc1)**2 + (yc-yc1)**2)

#min_dist = np.sum(np.min(dist, axis=1))

#return min_dist

#m = minimize(min_fun, [0.5])
#print(m.x[0]*math.pi, c1)

#m = minimize(min_fun, [c1])
#print(m, c1)

#cc = np.arange(1.08, 1.1, 0.000000001)
#dd = [ min_fun(ccc) for ccc in cc]
#mm = np.argmin(dd)
#print(dd[mm], mm, cc[mm], c1, min_fun(1.10751))
#break
#plt.show()