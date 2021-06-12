from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.widgets import TextBox
from super_queens import SuperQueens
import nqueens

phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l

def clean_data(radious, theta, r, f=True):
    radious = np.array(radious)
    radious = radious[np.where(radious >= r)]
    if f:
        theta = theta[-len(radious):]
    else:
        theta = theta[:len(radious)]

    theta = theta[-len(radious):]
    return radious, theta

def polar_2_spherical(theta, r, z0=1.0):  # polar 2 cartesian I think
    # FIX OF 12.01
    if len(np.where(r>z0)[0]) > 0:
        theta = theta[:-len(np.where(r>z0)[0])]
        r = r[:-len(np.where(r>z0)[0])]

    print('theta', theta.tolist())

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.sqrt(z0*z0 - r*r)

    '''
    theta = np.pi * np.arange(0, 1, 0.01)
    n = 7
    a = np.array([(math.pi*2*i)/n for i in range(n)])
    r = np.array([spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a])

    c = 2.0
    x = r*np.sin(theta)*np.cos(theta*c)
    y = r*np.sin(theta)*np.sin(theta*c)
    z = r*np.cos(theta)
    '''

    return x, y, z, theta, r

def rotate_points(cords, rotate):
    x, y, z = cords
    xd, yd, zd = rotate
    xn, yn, zn = cords

    if xd != 0:
        xn = x*math.cos(xd) - y*math.sin(xd)
        yn = x*math.sin(xd) + y*math.cos(xd)
        zn = z

    if yd != 0:
        xn = x*math.cos(yd) - z*math.sin(yd)
        zn = x*math.sin(yd) + z*math.cos(yd)
        yn = y

    if zd != 0:
        yn = y*math.cos(zd) - z*math.sin(zd)
        zn = y*math.sin(zd) + z*math.cos(zd)
        xn = x


    return xn, yn, zn

class Spiral:
    def __init__(self, c1, c2, sol):
        self.xd, self.yd, self.zd = 0, 0, 0
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        #self.ax = plt.axes(projection='3d')
        self.ax = plt.axes()
        #self.ax = plt.axes()
        n = len(sol)

        self.n, self.c1, self.c2, self.sol =  n, c1, c2, sol

        self.xi, self.yi, self.zi = 0,0,0
        self.x0, self.y0, self.z0 = 0.0, 0.0, 1.0

        self.X, self.Y, self.Z = 0,0,0
        self.delta = 0.05

        self.init_values = True

        self.f = [(2*i)/n for i in range(n)]

        self.a = np.array([(math.pi*2*i)/n for i in range(n)])
        print('a', self.a)

        self.r = np.array([spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in self.a])

    def press(self, event):

        if event.key == 'm':
            if self.delta == 0.01:
                self.delta = 0.05
            else:
                self.delta = 0.01

        if event.key == 't':
            self.xd = self.delta
        if event.key == 'e':
            self.xd = -self.delta

        if event.key == 'g':
            self.yd = self.delta
        if event.key == 'd':
            self.yd = -self.delta

        if event.key == 'r':
            self.zd = self.delta
        if event.key == 'v':
            self.zd = -self.delta

        self.X, self.Y, self.Z = self.rotate_points((self.X, self.Y, self.Z))
        self.fig.canvas.draw()
        plt.clf()
        self.draw3d()
        plt.draw()
        self.xd, self.yd, self.zd = 0, 0 ,0

    def rotate_points(self, cords):
        x, y, z = cords
        xn, yn, zn = x, y, z

        self.xi += self.xd
        self.yi += self.yd
        self.zi += self.zd
        #print("rot", self.xi, self.yi, self.zi)

        if self.xd != 0:
            xn = x*math.cos(self.xd) - y*math.sin(self.xd)
            yn = x*math.sin(self.xd) + y*math.cos(self.xd)
            zn = z
            self.x0 = self.x0*math.cos(self.xd) - self.y0*math.sin(self.xd)
            self.y0 = self.x0*math.sin(self.xd) + self.y0*math.cos(self.xd)

        if self.yd != 0:
            xn = x*math.cos(self.yd) - z*math.sin(self.yd)
            zn = x*math.sin(self.yd) + z*math.cos(self.yd)
            yn = y
            self.x0 = self.x0*math.cos(self.yd) - self.z0*math.sin(self.yd)
            self.z0 = self.x0*math.sin(self.yd) + self.z0*math.cos(self.yd)

        if self.zd != 0:
            yn = y*math.cos(self.zd) - z*math.sin(self.zd)
            zn = y*math.sin(self.zd) + z*math.cos(self.zd)
            xn = x
            self.y0 = self.y0*math.cos(self.zd) - self.z0*math.sin(self.zd)
            self.z0 = self.y0*math.sin(self.zd) + self.z0*math.cos(self.zd)


        return xn, yn, zn

    def get_center(self, x, y, c):
        b = math.log(c)

        # Points on spiral
        x1, x2 = x[100:102]
        y1, y2 = y[100:102]

        x3, x4 = x[150:152]
        y3, y4 = y[150:152]

        def get_equation(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            m = (y2 - y1)/(x2 - x1)
            b = y1 - m*x1
            return m, b

        m1, b1 = get_equation((x1, y1) , (x2, y2))
        m2, b2 = get_equation((x3, y3) , (x4, y4))

        t1 = 1/b
        m01 = (m1 - t1)/(m1*t1 + 1)
        b01 = y1 - m01*x1

        # Calculate radius line of second tangent line
        m02 = (m2 - t1)/(m2*t1 + 1)
        b02 = y3 - m02*x3

        # Calculate intersection between the two radius lines
        x0 = (b02 - b01)/(m01 - m02)
        y0 = m01*x0 + b01

        return x0, y0

    def draw3d(self):
        n, c1, c2, sol, a, r, f = self.n, self.c1, self.c2, self.sol, self.a, self.r, self.f

        an = [(math.pi*2*i)/n for i in range(n)]
        rn = [spiral1(c1, 10*math.pi + math.pi/n, aa) for aa in an]

        an = np.array(an)
        rn = np.array(rn)

        #self.ax = plt.axes(projection='3d')
        #self.ax = plt.axes()





        # Circles of principal
        for rr in r:
            theta = np.pi * np.arange(0, 2, 0.01)
            #X, Y, Z, _, _ = polar_2_spherical(theta, rr)
            #self.ax.plot3D(X, Y, [Z]*len(X))

        # Circles of secondary
        for rr in rn:
            theta = np.pi * np.arange(0, 2, 0.01)
            #X, Y, Z, _, _ = polar_2_spherical(theta, rr)
            #self.ax.plot3D(X, Y, [rr]*len(X))

        for aa, ff in zip(a, f):
            #self.draw_spiral((0, (12.01 - ff)*math.pi), c1, aa, 0, spiral1, 'b', False)

            #self.draw_spiral(( (2 - ff)*math.pi , (12.01 - ff)*math.pi), c1, aa, 0, spiral2, 'b', False)

            ts1 = [spiral1_inv(c1, t, aa) for t in r]
            #X, Y, Z, _, _ = polar_2_spherical(ts1, r)
            #self.ax.scatter3D(X, Y,  Z, s=10, c='y')

            ts2 = [spiral1_inv(c1, t, aa) for t in rn]
            #X, Y, Z; _, _ = polar_2_spherical(ts2, rn)
            #self.ax.scatter3D(X, Y, Z, s=10, c='g')

        self.draw_spiral((0, 12*math.pi), c2, a[5], 0, spiral1, 'blue', True)



        tt = [a[v] for v in sol]
        #X, Y, Z, _, _ = polar_2_spherical(tt, r)
        #self.ax.scatter3D(X, Y, Z, s=10, c='black')
        #d = [np.linalg.norm(np.array([0.0, 0.0, 1000]) - np.array([int(xi*1000), int(yi*1000), int(zi*1000)]))  for xi, yi, zi in zip(X, Y, Z) ]

        h = 3
        #self.ax.scatter([X[h]], [Y[h]], s=10, c='black')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface
        #self.ax.plot_surface(x, y, z, color=(0,0, 1, 0.1))

        '''
        if self.init_values:
            self.init_values = False
            self.X, self.Y, self.Z = self.draw_spiral((0, 12*math.pi), self.c2, 0 , 0, spiral1, 'g', True)

        self.X, self.Y, self.Z = self.draw_rot_spiral((0, 12*math.pi), self.c2, 0 , 0, spiral1, 'g', True)
        '''

        #self.ax.scatter3D(self.x0, self.y0, self.z0, s=10, c='b')

        #self.ax.view_init(elev=85, azim=100)


        #self.draw_spiral((0, 12*math.pi), self.c2, 0 , 0, spiral1, 'green', True, (5.0, -0.1, 0.05))
        #self.draw_spiral((0, 12*math.pi), self.c2, 0 , 0, spiral1, 'green', True, (math.pi*phi, -0.10, 0.05 ))
        #self.draw_spiral((0, 12*math.pi), self.c2, 0 , 0, spiral1, 'green', True, (math.pi*phi, -0.1, 0.05 ))

        #self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, e):
        print(e)



    def draw_spiral(self, point, c, aa, rr, spiral_func, color, show, origin=(0.0, 0.0, 0.0)):
        x0, y0, z0 = origin
        p1, p2 = point
        theta = np.arange(p1, p2, 0.001)
        radius = [spiral_func(c, t, aa) for t in theta]
        radius, theta = clean_data(radius, theta, rr)
        x, y, z, theta, radius = polar_2_spherical(theta, radius)

        #x = np.append([0.0], x)
        #y = np.append([0.0], y)
        #z = np.append([1.0], z)

        r = np.sqrt(x*x + y*y + z*z)
        t1 = np.arccos(x/r)
        t2 = np.arccos(y/np.sqrt(z*z + y*y))


        theta2 = np.arange(p1, p2*7, p2*7/len(theta))
        #x = radius*np.cos(theta)
        #y = radius*np.cos(theta + 2*math.pi/3)
        z = radius*np.cos(theta2)


        #print('t1', t1.tolist())
        #t2 = t1*2

        x1 = r*np.cos(t1)
        x2 = r*np.sin(t1)*np.cos(t2)
        x3 = r*np.sin(t1)*np.sin(t2)

        #t2calc = np.sin(t1)*np.cos()

        #plt.plot(theta, np.sqrt(z*z + y*y), c='r')
        #plt.plot(theta, y/np.sqrt(z*z + y*y), c='b')
        #plt.plot(theta, np.sqrt(x*x + y*y + z*z), c='purple')
        #plt.plot(theta, x, c='b')
        #plt.plot(theta, y, c='r')
        #plt.plot(theta, z, c='green')
        #plt.plot(theta,t2, c='b')

#x, y, z = rotate_points((x, y, z), (0.0, 0.0, z0))
        #x, y, z = rotate_points((x, y, z), (0.0, y0, 0.0))
        #x, y, z = rotate_points((x, y, z), (x0, 0.0, 0.0))

        l, = self.ax.plot(x, y, c=color)
        plt.subplots_adjust(bottom=0.2)
        #if show:
            #self.ax.plot(x, y, c=color)
            #self.ax.plot3D(x, y, z, c='b')
            #self.ax.plot3D(x, y, -np.sin(theta), c='r')
            #self.ax.plot3D(x1, x2, x3, c='red')
            #self.ax.plot(x, y, c=color)
            #self.ax.plot3D(x, y, z , c=color)

        def submit(text):
            #ydata = eval(text)
            #l.set_ydata(ydata)
            #self.ax.set_ylim(np.min(ydata), np.max(ydata))
            plt.draw()

            axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
            text_box = TextBox(axbox, 'Evaluate', initial='1.6')
            text_box.on_submit(submit)

        return x, y, z

ss = lambda x: int(x*1000)
def distancex1000_3d(p1, p2):
    x1, y1, z1 = p1
    x1, y1, z1 = ss(x1), ss(y1), ss(z1)
    x2, y2, z2 = p2
    x2, y2, z2 = ss(x2), ss(y2), ss(z2)

    return  (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)

from ortools.constraint_solver import pywrapcp

def find_center_3d():

    ds = [37179, 64547, 113501, 201897, 366084, 698581, 1997960]

    x1, y1, z1 = (-0.17305363748216343, 0.08333823950372914, 0.9813802404728816)
    x1, y1, z1 = ss(x1), ss(y1), ss(z1)
    x2, y2, z2 = (-0.299932653520666, -0.14443995328714523, 0.9429620900367413)
    x2, y2, z2 = ss(x2), ss(y2), ss(z2)
    x3, y3, z3 = (0.2732529304589817, 0.3426483369456679, 0.8988464569568859)
    x3, y3, z3 = ss(x3), ss(y3), ss(z3)

    solver = pywrapcp.Solver('LinearProgrammingExample')

    # Create the variables x and y.
    x = solver.IntVar(-1001, 1001, 'x')
    y = solver.IntVar(-1001, 1001, 'y')
    z = solver.IntVar(0, 1001, 'z')

    d1 = solver.IntVar([dd for dd in ds] , 'd1')
    d2 = solver.IntVar([dd for dd in ds], 'd2')
    d3 = solver.IntVar([dd for dd in ds], 'd3')

    solver.Add(solver.AllDifferent([d1, d2, d3]))
    solver.Add( ((x1-x)*(x1-x) + (y1-y)*(y1-y) + (z1-z)*(z1-z) ) == d1)
    solver.Add( ((x2-x)*(x2-x) + (y2-y)*(y2-y) + (z2-z)*(z2-z) ) == d2)
    solver.Add( ((x3-x)*(x3-x) + (y3-y)*(y3-y) + (z3-z)*(z3-z) ) == d3)

    q = [x, y,z, d1, d2, d3]
    db = solver.Phase(q, solver.CHOOSE_MIN_SIZE_LOWEST_MAX,solver.ASSIGN_CENTER_VALUE)

    solver.Solve(db)
    solver.NewSearch(db)

    while solver.NextSolution():
        r = [v.Value() for v in q]
        print(r)

find_center_3d()







'''
nq = nqueens.n_queens(7)
sols = nq.all_solutions

q = SuperQueens(7)
init_sols = [v.tolist() for v in q.init_sols()]

r = []
for s in sols:
    if s not in init_sols:
        r.append(s)

print(len(r), r)
'''


#12 [[3, 6, 4, 1, 5, 0, 2], [3, 0, 2, 5, 1, 6, 4], [2, 0, 5, 1, 4, 6, 3], [2, 6, 1, 3, 5, 0, 4], [4, 6, 1, 5, 2, 0, 3], [4, 0, 5, 3, 1, 6, 2], [1, 4, 6, 3, 0, 2, 5], [1, 4, 2, 0, 6, 3, 5], [1, 3, 0, 6, 4, 2, 5], [5, 2, 4, 6, 0, 3, 1], [5, 2, 0, 3, 6, 4, 1], [5, 3, 6, 0, 2, 4, 1]]

#s = Spiral(7, math.pow(phi, 2/(math.pi)), math.pow(phi, 1/(2*math.pi)), [4, 1, 5, 2, 6, 3, 0])
c1 = math.pow(phi, 2/math.pi)
c2 = math.pow(phi, 1/(math.pi*1))

#[3, 6, 4, 1, 5, 0, 2]

s = Spiral(c1, c2 , [3, 6, 4, 1, 5, 0, 2])
s.draw3d()
plt.show()



def rand_sign():
    return [-1,1][random.randrange(2)]


def rotate_points_mut(cords, rotation):
    xn, yn, zn = cords
    x, y, z = cords
    xd, yd, zd = rotation

    if zd != 0:
        yn = y*math.cos(zd) - z*math.sin(zd)
        zn = y*math.sin(zd) + z*math.cos(zd)
        xn = x

    if xd != 0:
        xn = x*math.cos(xd) - y*math.sin(xd)
        yn = x*math.sin(xd) + y*math.cos(xd)
        zn = z

    if yd != 0:
        xn = x*math.cos(yd) - z*math.sin(yd)
        zn = x*math.sin(yd) + z*math.cos(yd)
        yn = y

    return xn, yn, zn

def genetic_spiral(n, points):

    a = np.array([(math.pi*2*i)/n for i in range(n)])
    f = [(2*i)/n for i in range(n)]
    other_p = points
    c = math.pow(phi, 1/(1*math.pi))

    def get_score(point, c, aa, spiral_func, origin, other_points):
        p1, p2 = point
        x0, y0, z0 = origin
        theta = np.arange(p1, p2, 0.001)
        radius = [spiral_func(c, t, aa) for t in theta]
        radius, theta = clean_data(radius, theta, 0)
        x, y, z = polar_2_spherical(theta, radius)

        x, y, z = rotate_points_mut((x,y,z), (0.0, 0.0, z0))
        x, y, z = rotate_points_mut((x,y,z), (0.0, y0, 0.0))
        x, y, z = rotate_points_mut((x,y,z), (x0, 0.0, 0.0))

        pp = [np.array([v, b, u]) for v, b, u in zip(x, y, z)]

        total = 0
        for other_point in other_points:
            mm = list(map(lambda v: np.linalg.norm(v-other_point), pp))
            total += np.min(mm)

        return total

    def gen_ind():
        x, y, z, aa = 10*random.random()*rand_sign(), random.random()*rand_sign(), random.random()*rand_sign(), random.choice(a)
        score = get_score( (0, 12*math.pi - aa), c, aa, spiral1, (x, y, z), other_p)
        return [x, y, z, aa, score]

    def mutate(ind):
        copy_ind = [v for v in ind]
        pos = random.randint(0, 2)
        if pos == 0:
            copy_ind[pos] = copy_ind[pos]*random.random()*10
        else:
            copy_ind[pos] = copy_ind[pos]*random.random()
        score = get_score( (0, 12*math.pi - copy_ind[3]), c, copy_ind[3], spiral1, (copy_ind[0], copy_ind[1], copy_ind[2]), other_p)
        copy_ind[4] = score
        return copy_ind

    def cross(ind1, ind2):
        cut = random.randint(1,2)
        nind1 = ind1[:cut] + ind2[cut:]
        nind2 = ind2[:cut] + ind1[cut:]
        score1 = get_score( (0, 12*math.pi - nind1[3]), c, nind1[3], spiral1, (nind1[0], nind1[1], nind1[2]), other_p)
        score2 = get_score( (0, 12*math.pi - nind2[3]), c, nind2[3], spiral1, (nind2[0], nind2[1], nind2[2]), other_p)
        nind1[4] = score1
        nind2[4] = score2

        return nind1, nind2

    population_size = 100
    iterations = 10

    population = [gen_ind() for _ in range(population_size)]
    population.sort(key=lambda x: x[4])


    print("population ready")

    for e in range(iterations):
        print("it", e)
        for i in range(population_size):
            print(population[0])

            if random.random() < 0.05:
                p = mutate(population[i])
                population.append(p)

            if random.random() < 0.01:
                ind1, ind2 = cross(population[i], random.choice(population))
                population.append(ind1)
                population.append(ind2)

            if random.random() < 0.1:
                ind1 = gen_ind()
                population.append(ind1)

            population.sort(key=lambda x: x[4])
            population = population[:population_size]

    p00 = population[0]

    print(rotate_points_mut((0.0, 0.0, 1.0), (p00[0], p00[1], p00[2])))


#genetic_spiral(7, np.array([(-0.17305364, 0.08333824, 0.98138024), (-0.29993265, -0.14443995, 0.94296209), (0.27325293, 0.34264834, 0.89884646)]))

#from scipy.optimize import fsolve

x1, y1, z1 = (-0.17305364, 0.08333824, 0.98138024)
x2, y2, z2 = (-0.29993265, -0.14443995, 0.94296209)
x3, y3, z3 = (0.27325293, 0.34264834, 0.89884646)
c = math.pow(phi, 1/(math.pi*1))

def eval(p):
    xd, yd, zd, t1, t2, t3 = p

    def pu(t):
        r = math.pow(c, t - 12*math.pi)
        zn = r*math.sin(t)*math.sin(zd) + math.sqrt(1.0 - r*r)*math.cos(zd)
        yn = r*math.sin(t)*math.cos(zd) - math.sqrt(1.0 - r*r)*math.sin(zd)

        xn = r*math.cos(t)*math.cos(yd) - zn*math.sin(yd)
        zn = xn*math.sin(yd) + zn*math.cos(yd)

        xn = xn*math.cos(xd) - yn*math.sin(xd)
        yn = xn*math.sin(xd) + yn*math.cos(xd)

        return xn, yn, zn

    xn1, yn1, zn1 = pu(t1)
    xn2, yn2, zn2 = pu(t2)
    #xn3, yn3, zn3 = pu(t3)

    return (x1 - xn1, y1 - yn1, z1 - zn1, x2 - xn2, y2 - yn2, z2 - zn2) #, x3 - xn3, y3 - yn3, z3 - zn3)


#xd, yd, zd, t1, t2, t3 =  fsolve(eval, (5.5, -0.1, 0.05, 4*math.pi, 4*math.pi, 4*math.pi))

#print(xd, yd, zd, t1, t2, t3)

