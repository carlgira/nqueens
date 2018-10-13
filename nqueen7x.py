import numpy as np
import math
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def create_polar_graphs(n):

	phi = (1 + math.sqrt(5))/2
	c = pow(phi, 2/math.pi)
	c1 = math.pow(phi, 1/(2*math.pi)) # 7 OK
	#c1 = math.pow(phi, 2/(3*math.pi)) # 5 OK
	#c1 = math.pow(phi, 3/(5*math.pi)) # 6
	#c1 = math.pow(phi, 1/(3*math.pi)) # 9


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


	h = 0
	for theta, fraction in zip(angles, fractions):
		ax = plt.subplot2grid((1, 1), (0, 0), projection='polar')
		rsc_i = [math.pow(c, 2*math.pi/n - 2*math.pi + (2*math.pi/n)*i) for i in range(n)]
		thetasc_i = [math.log(rc, c1) + 12*math.pi - theta for rc in rsc_i]

		theta_i_g = np.pi * np.arange(0, 12 - fraction, 0.01)
		r_values_i_g = [spiral_func_i(t, c ,theta) for t in theta_i_g]
		rsc_i_g_l = [spiral_func_i(t, c1, theta) for t in theta_i_g]
		ax.plot(theta_i_g, r_values_i_g, c='b')
		ax.plot(theta_i_g, rsc_i_g_l, c='y')

		xc_i = rsc_i*np.cos(thetasc_i)
		yc_i = rsc_i*np.sin(thetasc_i)

		dist_i = np.sqrt((xc-xc_i)**2 + (yc-yc_i)**2)

		sol_i = np.argmin(dist_i, axis=0)
		min_dist_i = np.sum(np.min(dist_i, axis=0))
		print(sol_i, min_dist_i, is_valid(n, sol_i))
		if is_valid(n, sol_i) and sol_i.tolist() not in sols:
			sols.append(sol_i.tolist())

		rsc_d = [math.pow(c, -(2*math.pi/n)*(i)) for i in range(n)]
		thetasc_d = [math.log(rc, c1) - 2*math.pi + theta for rc in rsc_d]

		theta_d_g = np.arange(2*math.pi - theta, 10*math.pi, 0.01)
		r_values_d_g = [spiral_func_d(t, c, theta) for t in theta_d_g]
		rsc_d_g_l = [spiral_func_d(t, c1, theta) for t in theta_d_g]
		ax.plot(theta_d_g, r_values_d_g, c='g')
		ax.plot(theta_d_g, rsc_d_g_l, c='r')

		xc_d = rsc_d*np.cos(thetasc_d)
		yc_d = rsc_d*np.sin(thetasc_d)

		dist_d = np.sqrt((xc-xc_d)**2 + (yc-yc_d)**2)

		sol_d = np.argmin(dist_d, axis=0)
		min_dist_d = np.sum(np.min(dist_d, axis=0))
		print(sol_d, min_dist_d, is_valid(n, sol_d))
		if is_valid(n, sol_d) and sol_d.tolist() not in sols:
			sols.append(sol_d.tolist())

		ax.scatter(thetas.tolist(), rs.tolist(), color='black')
		ax.scatter(thetasc_i, rsc_i, color='red')
		ax.scatter(thetasc_d, rsc_d, color='green')


		h = h + 1

		plt.show()

	return sols



def is_valid(n, vec):
	cols = range(n)
	return n == len(set(vec)) and (n == len(set(vec[i]+i for i in cols)) == len(set(vec[i]-i for i in cols)))


o = create_polar_graphs(7)
print(o)




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