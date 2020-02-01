#Complex Sinus function  with coloring based to imaginary part
# Based on this comment http://stackoverflow.com/a/6543777
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
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


fig = plt.figure(figsize=(10, 10))
#ax = plt.subplot2grid((1, 1), (0, 0), projection='polar')
ax = fig.gca(projection='3d')

n = 7
c1 = math.pow(phi, 2/(math.pi))
f = [(2*i)/n for i in range(n)]

a = [(math.pi*2*i)/n for i in range(n)]
r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

aa = a[0]
ff = f[0]

theta1 = np.pi * np.arange(0, 12 - ff, 0.01)

X = np.array(theta1)
YY = np.arange(-2, 2, len(theta1))


X, Y = np.meshgrid(X, YY)

R=np.array([spiral1(c1, t + 1j*i, aa) for t,i in zip(theta1,YY)])
Z=R.real
T=R.imag
N = np.abs(T/T.max())  # normalize 0..1
plt.title(' $\mathrm{f(z)=sin(z)}$')
plt.xlabel(' $\mathrm{Re(z)}$')
plt.ylabel(' $\mathrm{Im(z)}$')
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    facecolors=cm.jet(N),
    linewidth=0, antialiased=True, shade=False)
# Colorbar see http://stackoverflow.com/a/6601210
m = cm.ScalarMappable(cmap=cm.jet, norm=surf.norm)
m.set_array(T)
p=plt.colorbar(m)
p.set_label(' $\mathrm{Im(f(z))}$')

fig.set_size_inches(14,7) #http://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
#plt.show() # if you run it as a python script
plt.show()

ax.plot(theta1, radious1, c='b')

'''

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-2*3.14, 2*3.14, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
R=np.sin(X + 1j*Y)
Z=R.real
T=R.imag
N = np.abs(T/T.max())  # normalize 0..1
plt.title(' $\mathrm{f(z)=sin(z)}$')
plt.xlabel(' $\mathrm{Re(z)}$')
plt.ylabel(' $\mathrm{Im(z)}$')
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    facecolors=cm.jet(N),
    linewidth=0, antialiased=True, shade=False)
# Colorbar see http://stackoverflow.com/a/6601210
m = cm.ScalarMappable(cmap=cm.jet, norm=surf.norm)
m.set_array(T)
p=plt.colorbar(m)
p.set_label(' $\mathrm{Im(f(z))}$')

fig.set_size_inches(14,7) #http://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
#plt.show() # if you run it as a python script
plt.show()
'''