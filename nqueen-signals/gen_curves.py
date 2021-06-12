import numpy as np
import math
import matplotlib.pyplot as plt

phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 2*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l


def polar_2_cartesian(theta, r, x0=0.0, y0=0.0):
    return r*np.cos(theta) + x0, r*np.sin(theta) + y0


class Spiral2D:
    def __init__(self):
        self.x0 = 0.0
        self.y0 = 0.0

    def draw(self, c1, sol):
        n = len(sol)
        f = [(2*i)/n for i in range(n)]

        a = [(math.pi*2*i)/n for i in range(n)]
        r = [spiral1(c1, math.pi*2/n, aa) for aa in a]

        an = [(math.pi*2*i)/n for i in range(n)]
        rn = [spiral1(c1, 10*math.pi + math.pi/n, aa) for aa in an]

        a = np.array(a)
        r = np.array(r)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot2grid((1, 1), (0, 0))

        for rr in r:
            theta = np.pi * np.arange(0, 2 + 0.01, 0.01)
            X, Y = polar_2_cartesian(theta, [rr]*len(theta), self.x0, self.y0)
            ax.plot(X, Y)

        for aa, ff in zip(a, f):
            theta1 = np.pi * np.arange(0, 12 - ff + 0.01, 0.01)
            radious1 = [spiral1(c1, t, aa) for t in theta1]

            theta2 = np.pi * np.arange(2 - ff, 12 + 0.01, 0.01)
            radious2 = [spiral2(c1, t, aa) for t in theta2]

            X, Y = polar_2_cartesian(theta1, radious1, self.x0, self.y0)
            #ax.scatter(X, Y, c='b')

            X, Y = polar_2_cartesian(theta2, radious2, self.x0, self.y0)
            #ax.scatter(X, Y, c='r')

        tt = [a[v] for v in sol]
        X0, Y0 = polar_2_cartesian(tt, r, self.x0, self.y0)
        ax.scatter(X0, Y0, s=30, c='black')


        plt.show()



spiral = Spiral2D()
spiral.draw(math.pow(phi, 2/(math.pi)), list(range(50)))
