from matplotlib import pyplot
import numpy as np
import nqueens
import math

n = 5
monitor = nqueens.n_queens(n)
print("asdds", monitor.all_solutions[0])
angle = math.pi*2/n


def transform(y, g=1):
	xr = [ float("{:.10f}".format((r+1)*math.cos(angle*i*g)))   for i,r in zip(y,range(len(y))) ]
	yr = [ float("{:.2f}".format((r+1)*math.sin(angle*i*g)))  for i,r in zip(y,range(len(y))) ]
	return xr, yr

def re_transform(y):
	xr = range(n)
	yr = [math]


sol = monitor.unique_solutions[1]
print(sol)
x,y = transform(sol)


print("(", x, "," ,y, ")")

r = [0.21455130690823604,0.3148694142107295, 0.4643,0.6795742752749557,1]

phi = (1 + math.sqrt(5))/2

c = pow(phi, 2/math.pi)

print("C", c, phi)

angle = math.pi*2/5

a = [math.pi*2/5, math.pi*4/5, math.pi*6/5, math.pi*8/5, math.pi*2]

ra = [math.pi*2, math.pi*8/5, math.pi*6/5, math.pi*4/5, math.pi*2/5]

theta = [math.log(rr)/math.log(c) + 12*math.pi - 2*math.pi/5 for rr in r]


#[0.21455130690823604,\ 0.3148694142107295,\ 0.4643,\ 0.6795742752749557,1]

r = [0.21455130690823604, 0.3148694142107295, 0.4643, 0.6795742752749557,1]

xx = [float("{:.10f}".format(rr*math.cos(a[t]))) for rr, t in zip(r, sol)]
yy = [float("{:.10f}".format(rr*math.sin(a[t]))) for rr, t in zip(r, sol)]


r1 = pow(c, math.pi*10 - 12*math.pi + math.pi*2/5)

r2 = pow(c, math.pi*10 - 12*math.pi + math.pi*4/5)

r3 = pow(c, math.pi*10 - 12*math.pi + math.pi*6/5)

r4 = pow(c, math.pi*10 - 12*math.pi + math.pi*8/5)

r5 = pow(c, math.pi*10 - 12*math.pi + math.pi*2)

rr = [r1, r2, r3, r4, r5]

for ssol in monitor.all_solutions:
	s = [f*a[g] for f,g in zip(rr, ssol)]
	print(sum(s), sum(s)/math.pi)




#t11 = math.log(y1)/math.log(c) + 12*math.pi - math.pi*8/5
#y11 = y1*math.sin(t11)
#x11 = y1*math.cos(t11)
#print(t11 , x11, y11)
