import numpy as np
from scipy.integrate import odeint, quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import seaborn as sbs
from matplotlib.widgets import TextBox
from matplotlib.pyplot import figure
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 20})
rc('text', usetex=True)
rc('animation', html='html5')

# The potential and its first derivative, as callables.
V = lambda x: 0.5 * x**2 * (0.5 * x**2 - 1)
dVdx = lambda x: x**3 - x

# The potential energy function on a grid of x-points.
xgrid = np.linspace(-1.5, 1.5, 100)
Vgrid = V(xgrid)

#plt.plot(xgrid, Vgrid)
#plt.xlabel('$x$')
#plt.ylabel('$V(x)$')

def deriv(X, t, gamma, delta, omega):
    """Return the derivatives dx/dt and d2x/dt2."""

    x, xdot = X
    xdotdot = -dVdx(x) -delta * xdot + gamma * np.cos(omega*t)
    return xdot, xdotdot

def solve_duffing(tmax, dt_per_period, t_trans, x0, v0, gamma, delta, omega):
    """Solve the Duffing equation for parameters gamma, delta, omega.

    Find the numerical solution to the Duffing equation using a suitable
    time grid: tmax is the maximum time (s) to integrate to; t_trans is
    the initial time period of transient behaviour until the solution
    settles down (if it does) to some kind of periodic motion (these data
    points are dropped) and dt_per_period is the number of time samples
    (of duration dt) to include per period of the driving motion (frequency
    omega).

    Returns the time grid, t (after t_trans), position, x, and velocity,
    xdot, dt, and step, the number of array points per period of the driving
    motion.

    """
    # Time point spacings and the time grid

    period = 2*np.pi/omega
    dt = 2*np.pi/omega / dt_per_period
    step = int(period / dt)
    t = np.arange(0, tmax, dt)
    # Initial conditions: x, xdot
    X0 = [x0, v0]
    X = odeint(deriv, X0, t, args=(gamma, delta, omega))
    idx = int(t_trans / dt)
    return t[idx:], X[idx:], dt, step




# Set up the motion for a oscillator with initial position
# x0 and initially at rest.
x0, v0 = 0, 0
tmax, t_trans = 50, 1
omega = 1.4
gamma, delta = 0.39, 0.1
dt_per_period = 100

t, X, dt, pstep = solve_duffing(tmax, dt_per_period, t_trans, x0, v0, gamma, delta, omega)
x, xdot = X.T

# The animation
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 9))

# Position as a function of time
ax2 = ax[0]
ax2.set_xlabel(r'$t / \mathrm{s}$')
ax2.set_ylabel(r'$x / \mathrm{m}$')
ln2, = ax2.plot(t, x)
ax2.set_ylim(np.min(x), np.max(x))

# Phase space plot
ax3 = ax[1]
ax3.set_xlabel(r'$x / \mathrm{m}$')
ax3.set_ylabel(r'$\dot{x} / \mathrm{m\,s^{-1}}$')
ln3, = ax3.plot([], [])
ax3.set_xlim(-2, 2)
ax3.set_ylim(np.min(xdot), np.max(xdot))

# Poincaré section plot
ax4 = ax[2]
ax4.set_xlabel(r'$x / \mathrm{m}$')
ax4.set_ylabel(r'$\dot{x} / \mathrm{m\,s^{-1}}$')
ax4.scatter(x[::pstep], xdot[::pstep], s=2, lw=0, c=sbs.color_palette()[0])
#plt.tight_layout()

ln2.set_data(t, x)
ln3.set_data(x, xdot)

def update():
    global t, X, dt, pstep, X, xdot, ln1, ln2, ln3
    t, X, dt, pstep = solve_duffing(tmax, dt_per_period, t_trans, x0, v0, gamma, delta, omega)
    x, xdot = X.T

    ln2.set_data(t, x)
    ln3.set_data(x, xdot)

axbox_x0 = plt.axes([0.1, 0.20, 0.1, 0.05])
x0_text_box = TextBox(axbox_x0, 'x0', initial=str(x0))
def setx0(x): global x0; x0 = float(x); update();
x0_text_box.on_submit(setx0)

axbox_v0 = plt.axes([0.3, 0.20, 0.1, 0.05])
v0_text_box = TextBox(axbox_v0, 'v0', initial=str(v0))
def setv0(x): global v0; v0 = float(x); update();
v0_text_box.on_submit(setv0)

axbox_omega = plt.axes([0.6, 0.20, 0.3, 0.05])
omega_text_box = TextBox(axbox_omega, 'omega', initial=str(omega))
def setOmega(x): global omega; omega = float(x); update();
omega_text_box.on_submit(setOmega)

axbox_gamma = plt.axes([0.1, 0.25, 0.3, 0.05])
gamma_text_box = TextBox(axbox_gamma, 'gamma', initial=str(gamma))
def setGamma(x): global gamma; gamma = float(x); update();
gamma_text_box.on_submit(setGamma)

axbox_delta = plt.axes([0.5, 0.25, 0.3, 0.05])
delta_text_box = TextBox(axbox_delta, 'delta', initial=str(delta))
def setDelta(x): global delta; delta = float(x); update();
delta_text_box.on_submit(setDelta)


plt.subplots_adjust(left=0.1, bottom=0.40)
plt.show()