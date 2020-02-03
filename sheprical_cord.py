from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

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

def polar_2_spherical(theta, r):
    # FIX OF 12.01
    if len(np.where(r>1)[0]) > 0:
        r[np.where(r>1)] = 1.0
    return r*np.cos(theta) , r*np.sin(theta) , np.sqrt(1 - r*r)

class Spiral:
    def __init__(self, n, c1, c2, sol):
        self.xd, self.yd, self.zd = 0, 0, 0
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        self.n, self.c1, self.c2, self.sol =  n, c1, c2, sol

        self.xi, self.yi, self.zi = math.pi/4, math.pi/4, math.pi/4

    def press(self, event):

        if event.key == 'r':
            self.xd = 0.1
        if event.key == 'f':
            self.xd = 0.1

        if event.key == 'd':
            self.yd = 0.1
        if event.key == 'g':
            self.yd = 0.1

        if event.key == 'e':
            self.zd = 0.1
        if event.key == 't':
            self.zd = 0.1

        self.fig.canvas.draw()
        plt.clf()
        self.draw3d()
        plt.draw()
        self.xd, self.yd, self.zd = 0.001, 0.001, 0.001

    def rotate_points(self, cords):
        x, y, z = cords
        xn, yn, zn = x, y, z

        self.xi += self.xd
        self.yi += self.yd
        self.zi += self.zd

        if self.xd != 0:
            xn = x*math.cos(self.xi) + y*math.sin(self.xi)
            yn = -x*math.sin(self.xi) + y*math.cos(self.xi)
            zn = z

        if self.yd != 0:
            xn = x*math.cos(self.yi) - z*math.sin(self.yi)
            zn = x*math.sin(self.yi) + + z*math.cos(self.yi)
            yn = y

        if self.zd != 0:
            yn = y*math.cos(self.zi) - z*math.sin(self.zi)
            zn = y*math.sin(self.zi) + + z*math.cos(self.zi)
            xn = x

        return xn, yn, zn

    def draw3d(self):
        n, c1, c2, sol = self.n, self.c1, self.c2, self.sol

        f = [(2*i)/n for i in range(n)]

        a = [(math.pi*2*i)/n for i in range(n)]
        r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

        an = [(math.pi*2*i)/n for i in range(n)]
        rn = [spiral1(c1, 10*math.pi + math.pi/n, aa) for aa in an]

        a = np.array(a)
        r = np.array(r)

        an = np.array(an)
        rn = np.array(rn)

        ax = plt.axes(projection='3d')

        # Circles of principal
        for rr in r:
            theta = np.pi * np.arange(0, 2, 0.01)
            X, Y, Z = polar_2_spherical(theta, rr)
            #ax.plot3D(X, Y, [Z]*len(X))

        # Circles of sencondary
        for rr in rn:
            theta = np.pi * np.arange(0, 2, 0.01)
            X, Y, Z = polar_2_spherical(theta, rr)
            #ax.plot3D(X, Y, [rr]*len(X))

        for aa, ff in zip(a, f):
            theta1 = np.pi * np.arange(0, 12.01 - ff, 0.01)
            radious1 = [spiral1(c1, t, aa) for t in theta1]
            radious1, theta1 = clean_data(radious1, theta1, r[0])

            theta2 = np.pi * np.arange(2 - ff, 12.01, 0.01)
            radious2 = [spiral2(c1, t, aa) for t in theta2]
            radious2, theta2 = clean_data(radious2, theta2, r[0], False)

            X, Y, Z = polar_2_spherical(theta1, radious1)
            #ax.plot3D(X, Y, Z, c='b')

            X, Y, Z = polar_2_spherical(theta2, radious2)
            #x.plot3D(X, Y, Z, c='b')

            ts1 = [spiral1_inv(c1, t, aa) for t in r]
            X, Y, Z = polar_2_spherical(ts1, r)
            #ax.scatter3D(X, Y,  Z, s=10, c='y')

            ts2 = [spiral1_inv(c1, t, aa) for t in rn]
            X, Y, Z = polar_2_spherical(ts2, rn)
            #ax.scatter3D(X, Y, Z, s=10, c='g')

        theta3 = np.pi * np.arange(0, 12.01 - f[0], 0.01)
        radious3 = [spiral1(c2, t, a[0]) for t in theta3]
        radious3, theta3 = clean_data(radious3, theta3, r[0])

        theta4 = np.pi * np.arange(2 - f[0], 12.01, 0.01)
        radious4 = [spiral2(c2, t, a[0]) for t in theta4]
        radious4, theta4 = clean_data(radious4, theta4, r[0], False)

        X, Y, Z = polar_2_spherical(theta3, radious3)
        ax.plot3D(X, Y, Z , c='black')

        DX, DY, DZ = self.rotate_points((X, Y, Z))
        ax.plot3D(DX, DY, DZ , c='red')

        X, Y, Z = polar_2_spherical(theta4, radious4)
        #ax.plot3D(X, Y, Z , c='gray')

        tt = [a[v] for v in sol]
        X, Y, Z = polar_2_spherical(tt, r)
        ax.scatter3D(X, Y, Z, s=20, c='black')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the surface
        ax.plot_surface(x, y, z, color=(0,0, 1, 0.5))

s = Spiral(7, math.pow(phi, 2/(math.pi)), math.pow(phi, 1/(2*math.pi)), [4, 1, 5, 2, 6, 3, 0])
s.draw3d()
plt.show()