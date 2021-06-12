import numpy as np
import matplotlib.pyplot as plt
import math

# Spiral definition
a = 0.5
b = 0.20
th = np.linspace(0, 6*math.pi, 1000)
x0, y0 = 2.5, 3.7 # Initial center of spiral
x = a*np.exp(b*th)*np.cos(th) + x0
y = a*np.exp(b*th)*np.sin(th) + y0

plt.plot(x, y)

# Points on spiral
x1, x2 = x[500:502]
y1, y2 = y[500:502]

x3, x4 = x[800:802]
y3, y4 = y[800:802]

plt.scatter([x1, x3], [y1, y3], s=10, c='b'  )

# Calculate slope and intercept using two points
def get_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    return m, b

# Plot line function
def plot_line(x1, x2, m, b, c='red'):
    x = np.linspace(x1, x2, 100)
    y = m*x + b
    plt.plot(x, y, c=c)

# Tangent equations of points in spiral
m1, b1 = get_equation((x1, y1) , (x2, y2))
m2, b2 = get_equation((x3, y3) , (x4, y4))

plot_line(-1, -0.5, m1, b1)
plot_line(-7, -4, m2, b2)

# Calculate radius line of first tangent line
t1 = 1/b
m01 = (m1 - t1)/(m1*t1 + 1)
b01 = y1 - m01*x1
plot_line(-1, 3, m01, b01, c='g')

# Calculate radius line of second tangent line
m02 = (m2 - t1)/(m2*t1 + 1)
b02 = y3 - m02*x3
plot_line(-7, 3, m02, b02, c='g')

# Calculate intersection between the two radius lines
x0 = (b02 - b01)/(m01 - m02)
y0 = m01*x0 + b01
print(x0, y0)
plt.scatter([x0], [y0], s=20, c='black' , zorder=0 )

plt.show()