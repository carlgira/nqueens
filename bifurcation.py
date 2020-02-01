import numpy as np
import matplotlib.pyplot as plt


def logistic(r, x):
    return r * x*(1 - x)

n = 10000
r = np.linspace(2.5, 3.57, n)

iterations = 1000
last = 100

x = 1e-5 * np.ones(n)

fig, (ax1) = plt.subplots(1, 1, figsize=(8, 9), sharex=True)
for i in range(iterations):
    x = logistic(r, x)
    # We compute the partial sum of the
    # Lyapunov exponent.
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        ax1.plot(r, x, ',k', alpha=.25)
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")


plt.tight_layout()


plt.show()