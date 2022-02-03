import numpy as np
import math
import matplotlib.pyplot as plt

# Acceleration due to gravity (m.s-2); final position of bead (m).
x1, y1 = 1, 1
x2, y2 = 2, 1.65
N = 100
theta = np.linspace(0, 2*np.pi, N)

eps = np.finfo(float).eps

def get_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    return m, b

def get_perpendicular(m, b, p1):
    x1, y1 = p1
    m1 = -1/m
    b1 = y1 - x1*m1

    return m1, b1

def intersection_btw_lines(m1, b1, m2, b2):
    x = (b2 - b1)/(m1 - m2)
    y = m1*x + b1
    return x, y

def cycloid(p1, p2, N=100):
    x1, y1 = p1
    x2, y2 = p2

    d = np.linalg.norm(np.array(p1)-np.array(p2))
    r = d/(2*np.pi)

    theta = np.linspace(0, np.pi, N)
    x = r * (theta - np.sin(theta))
    y = r * (1 - np.cos(theta))

    plt.plot(x,y)
    plt.show()

    s = 1 if y2 > y1 else -1
    tr = s*(math.acos(abs(x2-x1)/d) + math.pi)

    xr = y*np.cos(theta)
    yr = y*np.sin(theta)

    #xr = math.cos(tr)*x - math.sin(tr)*y + x2
    #yr = math.sin(tr)*x + math.cos(tr)*y + y2

    return xr, yr


def convert_arc(pt1, pt2, sagitta):
    x1, y1 = pt1
    x2, y2 = pt2

    # find normal from midpoint, follow by length sagitta
    n = np.array([y2 - y1, x1 - x2])
    n_dist = np.sqrt(np.sum(n**2))

    if sagitta is None:
        sagitta = n_dist/2
    else:
        sagitta = n_dist * sagitta

    if np.isclose(n_dist, 0):
        # catch error here, d(pt1, pt2) ~ 0
        print('Error: The distance between pt1 and pt2 is too small.')

    n = n/n_dist
    x3, y3 = (np.array(pt1) + np.array(pt2))/2 + sagitta * n
    #ax.scatter([x3], [y3])

    # calculate the circle from three points
    # see https://math.stackexchange.com/a/1460096/246399
    A = np.array([
        [x1**2 + y1**2, x1, y1, 1],
        [x2**2 + y2**2, x2, y2, 1],
        [x3**2 + y3**2, x3, y3, 1]])
    M11 = np.linalg.det(A[:, (1, 2, 3)])
    M12 = np.linalg.det(A[:, (0, 2, 3)])
    M13 = np.linalg.det(A[:, (0, 1, 3)])
    M14 = np.linalg.det(A[:, (0, 1, 2)])

    if np.isclose(M11, 0):
        # catch error here, the points are collinear (sagitta ~ 0)
        print('Error: The third point is collinear.')

    cx = 0.5 * M12/M11
    cy = -0.5 * M13/M11
    radius = np.sqrt(cx**2 + cy**2 + M14/M11)

    # calculate angles of pt1 and pt2 from center of circle
    pt1_angle = 180*np.arctan2(y1 - cy, x1 - cx)/np.pi
    pt2_angle = 180*np.arctan2(y2 - cy, x2 - cx)/np.pi

    return (cx, cy), radius, pt1_angle, pt2_angle


def circle(pt1, pt2, sagitta=None):
    center, radius, start_angle, end_angle = convert_arc(pt1, pt2, sagitta)
    x0, y0 = center

    theta = []

    x1, y1 = pt1
    x2, y2 = pt2

    if y2 > y1:
        theta = np.linspace(end_angle*np.pi/180, start_angle*np.pi/180, N)
    else:
        if start_angle < 0:
            start_angle += 360

        if end_angle < 0:
            end_angle += 360
        theta = np.linspace(start_angle*np.pi/180, end_angle*np.pi/180, N)

    x = radius*np.cos(theta) + x0
    y = radius*np.sin(theta) + y0

    if theta[0] > theta[-1]:
        x = np.flip(x)
        y = np.flip(y)

    return x, y


def orbit_smooth(r1, r2):
    radious = abs(r1-r2)
    # Line
    #r = np.linspace(r1, r2, N)

    # Circle
    #xx = np.linspace(0, radious, N)
    #r = -np.sqrt(radious * radious - xx * xx) + r2
    #if r1 > r2:
    #    r = -r

    # Real
    xx = np.linspace(-1, 0.5, N)
    r = radious/2*np.tanh(4*xx)
    if r1 > r2:
        r = -r
    r += -np.min(r)

    #plt.scatter(xx, r)
    #plt.show()

    x = r*np.cos(theta) + r1
    y = r*np.sin(theta) + r1
    return x, y


def orbit(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    theta = np.linspace(0, 2*np.pi, N)
    x = (x2+1)*np.cos(theta) - x2 + 1
    y = (x2+1)*np.sin(theta)

    return x, y

orbit_flag = True
def half_orbit(p1, p2):
    global orbit_flag
    x1, y1 = p1
    x2, y2 = p2

    theta = np.linspace(0, np.pi, N)
    if not orbit_flag:
        theta = np.linspace(np.pi, 2*np.pi, N)

    x = (x2+1)*np.cos(theta) - x2 + 1
    y = (x2+1)*np.sin(theta)

    orbit_flag = not(orbit_flag)

    return x, y

def spiral(p1, p2, N=100):
    x1, y1 = p1
    x2, y2 = p2

    b = (x2 - x1)/(2*np.pi)
    x = (x1 + b*theta)*np.cos(theta)
    y = (x1 + b*theta)*np.sin(theta)

    return x, y

spiral_flag = True
def half_spiral(p1, p2, N=100):
    global spiral_flag
    x1, y1 = p1
    x2, y2 = p2

    if spiral_flag:
        x2 = x2*-1
    else:
        x1 = x1*-1

    b = (x1 - x2)/(np.pi)
    theta = np.linspace(0, np.pi, 100)
    x = (x1 + b*theta)*np.cos(theta)
    y = (x1 + b*theta)*np.sin(theta)

    spiral_flag = not(spiral_flag)

    return x, y

def helix(p1, p2, N=100):
    x1, y1 = p1
    x2, y2 = p2
    m = 0.01

    x1 += 1
    x2 += 1

    signal = -1
    if x1 > x2:
        signal = 1

    theta = np.linspace(0, 2*np.pi, N)
    r = np.geomspace(signal*m, signal*abs(x2-x1), N) + (x1 - m)
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return x, y


def circular_motion(n, sol, f, r, t=1, v0=0, points=100):
    a = [f*(i+1) for i in range(n)]
    print(a)
    print(np.sin(a).tolist())

    y = []
    y.append(a[sol[0]])
    yf = a[sol[0]]
    for i in range(len(sol)-1):
        if a[sol[i+1]] > a[sol[i]]:
            yf = a[sol[i+1]]
        else:
            yf = 2*np.pi - (a[sol[i]] - a[sol[i+1]])

        acc = (yf - y[-1] - v0*t)*2/t*t
        time = np.linspace(0.1, 1, points)
        yy = 0.5*acc*time*time + v0*time + y[-1]
        y.extend(yy.tolist())
        v0 = acc*t + v0

    return y, r*np.sin(y)


def gg(t):
    x = np.linspace(0, 10, t)
    y = np.geomspace(-0.01, -10, t)
    plt.plot(x, y)
    plt.show()


def orbit_spiral(r):

    x_all = np.array([])
    y_all = np.array([])
    for i in range(len(r)-1):
        x, y = spiral((r[i],0), (r[i+1],0))
        if i != len(r)-2:
            x_all = np.concatenate([x_all, x[:-1]])
            y_all = np.concatenate([y_all, y[:-1]])
        else:
            x_all = np.concatenate([x_all, x])
            y_all = np.concatenate([y_all, y])

    return x_all, y_all



'''
fig, ax = plt.subplots()

p1 = (x1, y1)
p2 = (x2, y2)

sa = 0.5
sol = [2, 5, 1, 4, 0]
oo = 0.0001

shape_fun = half_orbit

p1 = (sol[0], 0)
p2 = (sol[1], 0)
x, y = shape_fun(p1, p2)
ax.plot(x, y, lw=4, alpha=0.5)

p1 = (sol[1], 0)
p2 = (sol[2], 0)
x, y = shape_fun(p1, p2)
ax.plot(x, y, lw=4, alpha=0.5)

p1 = (sol[2], 0)
p2 = (sol[3], 0)
x, y = shape_fun(p1, p2)
ax.plot(x, y, lw=4, alpha=0.5)

p1 = (sol[3], 0)
p2 = (sol[4], 0)
x, y = shape_fun(p1, p2)
ax.plot(x, y, lw=4, alpha=0.5)


#for sagitta in [0.05, 0.1, 0.5, 1]:#, 1, 2]: #1.5, 2, 0.5]:
#    x, y = circle(p1, p2, sagitta)
#    ax.plot(x, y, lw=4, alpha=0.5)

plt.show()
'''

'''
n = 5
points=100
fig, ax = plt.subplots()
x, y = circular_motion(n, [1, 4, 2, 3], 2*np.pi/n, 1, points=points)
print(x)
print('x', x[::points], np.sin(x[::points]), np.arcsin(np.sin(x[::points])))
print('y', y[::points])
ax.plot(np.linspace(0, 4*2*np.pi/5, len(y)), y)
plt.show()
#[2.5, 6.2, 3.7, 5.0]
'''


'''
r =[0.1, 0.3, 0.5,  0.2, 0.1]
v = 0
x_all = np.array([])
y_all = np.array([])
for i in range(len(r)-1):
    x, y = spiral((r[i],0), (r[i+1],0))
    if i != len(r)-2:
        x_all = np.concatenate([x_all, x[:-1]])
        y_all = np.concatenate([y_all, y[:-1]])
    else:
        x_all = np.concatenate([x_all, x])
        y_all = np.concatenate([y_all, y])
    plt.scatter(x,y)

plt.plot(x_all, y_all)
plt.show()

plt.plot(x_all)
plt.plot(y_all)
plt.plot(y_all + x_all)
plt.show()
'''
