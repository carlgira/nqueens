import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import math
import sys

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(sys.float_info.epsilon, 12 * math.pi, 0.001)


def polar_2_spherical(theta, r, z0=1.0):  # polar 2 cartesian I think

    # FIX OF 12.01
    if len(np.where(r > z0)[0]) > 0:
        theta = theta[:-len(np.where(r > z0)[0])]
        r = r[:-len(np.where(r > z0)[0])]

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(z0 * z0 - r * r)

    return x, y, z, theta, r


phi = (1 + math.sqrt(5)) / 2
spiral1 = lambda c, t, l: math.pow(c, t - 12 * math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12 * math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2 * math.pi + l))
spiral2_inv = lambda c, r, l: 2 * math.pi - math.log(r, c) - l

n = 7
c1 = math.pow(phi, 2 / math.pi)
c2 = math.pow(phi, 1 / (math.pi * 1))
a = np.array([(math.pi * 2 * i) / n for i in range(n)])
radius = np.array([spiral1(c1, tt, a[5]) for tt in t])

x, y, z, t, radius = polar_2_spherical(t, radius)


t1 = np.arange(sys.float_info.epsilon, math.pi/2, 0.001)
t = np.arange(sys.float_info.epsilon, 12 * math.pi, 12 * math.pi/len(t1))
r = np.array([spiral1(c1, tt, a[5]) for tt in t])
t0 = np.arcsin(r)
c = 7

x1 = np.sin(t0)*np.cos(c*t1)
x2 = np.sin(t0)*np.sin(c*t1)
#x3 = np.cos(t1)

#ax = plt.axes(projection='3d')
plt.plot(x1, x2, c='red')
#plt.plot(x, y, c='blue')
#plt.plot(t, a, c='blue')



def submit(text):
    ydata = eval(text)
    #l.set_ydata(ydata)
    # ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

initial_text='t1'
axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
text_box.on_submit(submit)

plt.show()
