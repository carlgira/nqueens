import nqueens as nq
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

n = 5
monitor = nq.n_queens(n)

sol = monitor.unique_solutions[0]

T = 2*math.pi

sep = 2/(n+1)
sep_angle = T/(n+1)

print(sol)

vr = [math.sqrt(1 - (1 - (sep*i))**2) for i in range(1, n+1)]
vz = [1 - (sep*i) for i in range(1, n+1)]


Y = [vr[x]*math.sin((x+1)*sep_angle) for x in sol]
X = [vr[x]*math.cos((x+1)*sep_angle) for x in range(n)]
Z = [vz[x] for x in range(0, n)]



fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z)

#draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="r")


pyplot.show()


