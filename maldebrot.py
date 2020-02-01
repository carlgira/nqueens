import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# counts the number of iterations until the function diverges or
# returns the iteration threshold that we check until
def countIterationsUntilDivergent(c, threshold):
    z = complex(0, 0)
    for iteration in range(threshold):
        z = (z*z) + c

        if abs(z) > 4:
            break
            pass
        pass
    return iteration

# takes the iteration limit before declaring function as convergent and
# takes the density of the atlas
# create atlas, plot mandelbrot set, display set
def mandelbrot(threshold, density):
    # location and size of the atlas rectangle
    # realAxis = np.linspace(-2.25, 0.75, density)
    # imaginaryAxis = np.linspace(-1.5, 1.5, density)
    realAxis = np.linspace(-0.22, -0.219, 1000)
    imaginaryAxis = np.linspace(-0.70, -0.699, 1000)
    realAxisLen = len(realAxis)
    imaginaryAxisLen = len(imaginaryAxis)

    # 2-D array to represent mandelbrot atlas
    atlas = np.empty((realAxisLen, imaginaryAxisLen))

    # color each point in the atlas depending on the iteration count
    for ix in range(realAxisLen):
        for iy in range(imaginaryAxisLen):
            cx = realAxis[ix]
            cy = imaginaryAxis[iy]
            c = complex(cx, cy)

            atlas[ix, iy] = countIterationsUntilDivergent(c, threshold)
            pass
        pass

    # plot and display mandelbrot set
    plt.imshow(atlas.T, interpolation="nearest")
    plt.show()

# time to party!!
#mandelbrot(120, 1000)


def mandelbrot_points(z=complex(0, 0), c=complex(0, 0), l=np.array([])):

    z = (z*z) + c
    if (len(l) > 0 and round(abs(z), 2) == round(abs(l[-1]), 2)) or len(l) > 200:
        print(len(l), round(abs(z), 2), round(abs(l[-1]), 2))
        return l

    return mandelbrot_points(z, c, np.append(l, z))


fig = plt.figure(figsize=(15, 15))

st = 10
x = np.linspace(0.45, 0.55, st)
for i in range(st):
    for j in range(st):
        ax = plt.subplot2grid((st, st), (i, j))
        ax.set_title( str([[x[i]], x[j]]), va='bottom', fontdict = {'fontsize': 10})

        xy = mandelbrot_points(c=complex(-x[i],x[j]))
        X = [x.real for x in xy]
        Y = [x.imag for x in xy]

        plt.scatter(X,Y, color='red')

plt.show()