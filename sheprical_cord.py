from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math



# Data for a three-dimensional line
'''
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

plt.show()
'''

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
    return r*np.cos(theta), r*np.sin(theta), theta

def draw3d(n, c1, c2, sol):

    f = [(2*i)/n for i in range(n)]

    a = [(math.pi*2*i)/n for i in range(n)]
    r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

    an = [(math.pi*2*i)/n for i in range(n)]
    rn = [spiral1(c1, 10*math.pi + math.pi/n, aa) for aa in an]

    a = np.array(a)
    r = np.array(r)

    an = np.array(an)
    rn = np.array(rn)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Circles of principal
    for rr in r:
        theta = np.pi * np.arange(0, 2, 0.01)
        X, Y, Z = polar_2_spherical(theta, rr)
        ax.plot3D(X, Y, Z)

    # Circles of sencondary
    for rr in rn:
        theta = np.pi * np.arange(0, 2, 0.01)
        X, Y, Z = polar_2_spherical(theta, rr)
        #ax.plot3D(X, Y, Z)

    for aa, ff in zip(a, f):
        theta1 = np.pi * np.arange(0, 12 - ff, 0.01)
        radious1 = [spiral1(c1, t, aa) for t in theta1]
        radious1, theta1 = clean_data(radious1, theta1, r[0])

        theta2 = np.pi * np.arange(2 - ff, 12, 0.01)
        radious2 = [spiral2(c1, t, aa) for t in theta2]
        radious2, theta2 = clean_data(radious2, theta2, r[0], False)

        X, Y, Z = polar_2_spherical(theta1, radious1)
        #ax.plot3D(X, Y, Z, c='b')

        X, Y, Z = polar_2_spherical(theta2, radious2)
        #ax.plot3D(X, Y, Z, c='b')

        ts1 = [spiral1_inv(c1, t, aa) for t in r]
        X, Y, Z = polar_2_spherical(ts1, r)
        #ax.scatter3D(X, Y,  Z, s=20, c='y')

        ts2 = [spiral1_inv(c1, t, aa) for t in rn]
        X, Y, Z = polar_2_spherical(ts2, rn)
        #ax.scatter3D(X, Y, Z, s=20, c='g')

    theta3 = np.pi * np.arange(0, 12 - f[0], 0.01)
    radious3 = [spiral1(c2, t, a[0]) for t in theta3]
    radious3, theta3 = clean_data(radious3, theta3, r[0])

    theta4 = np.pi * np.arange(2 - f[0], 12, 0.01)
    radious4 = [spiral2(c2, t, a[0]) for t in theta4]
    radious4, theta4 = clean_data(radious4, theta4, r[0], False)

    X, Y, Z = polar_2_spherical(theta3, radious3)
    ax.plot3D(X, Y, Z , c='black')

    X, Y, Z = polar_2_spherical(theta4, radious4)
    ax.plot3D(X, Y, Z , c='gray')

    tt = [a[v] for v in sol]
    X, Y, Z = polar_2_spherical(tt, r)
    ax.scatter3D(X, Y, Z, s=30, c='black')


    plt.show()

draw3d(7, math.pow(phi, 2/(math.pi)), math.pow(phi, 1/(2*math.pi)), [4, 1, 5, 2, 6, 3, 0])