import random
import numpy as np
import matplotlib.pyplot as plt
import math
import curves
import nqueens


n = 7
mutate_rate = 0.1
phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l
c1 = math.pow(phi, 2/(math.pi))

a = [(math.pi*2*i)/n for i in range(n)]
rr = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]


def fourier_series_coeff_numpy(x, return_complex=False):
    y = np.fft.rfft(x) / len(x)
    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag


def random_sign():
    return 1 if random.random() < 0.5 else -1


def build_fun(sol):
    '''
    y = []
    for i in range(len(sol)-1):
        s1 = sol[i]
        s2 = sol[i+1]
        xx, yy = curves.half_orbit((s1, 0), (s2, 0))
        y.extend(yy)

    x = np.linspace(0, np.pi*len(sol), len(y))

    return x, np.array(y)
    '''

    tt = [rr[v] for v in sol]
    x,y = curves.orbit_spiral(tt)

    return x,y


def decode_wave(wave):
    ym = np.abs(np.fft.irfft(wave.e))
    values = []
    flag = True
    for i in range(len(ym) -1):
        if flag and ym[i] > ym[i+1]:
            flag = False
            values.append(ym[i])
            continue
        if not flag and ym[i] < ym[i+1]:
            flag = True
            values.append(ym[i])

    ys = np.round(values) - 1

    return ys[ys >= 0]

def wave_from_fourier_coff(a0, an, bn):
    x = np.linspace(0, 2*np.pi, 85*7)
    y = a0 + an[0]*np.cos(x) + bn[0]*np.sin(x)

    o = 0
    for i, (a,b) in enumerate(zip(an[1:49], bn[1:-49])):
        y = y + a*np.cos(x*(i+2)) + b*np.sin(x*(i+2))
        o = i

    #print(o, len(an))

    return x,y


class Wave:
    def __init__(self, n, e):
        self.n = n
        self.e = e
        self.strength = len(e)

    def clone(self):
        wave = Wave(self.n, np.copy(self.e))
        return wave

    def mutate(self):
        wave = self.clone()
        coin = random.random()
        index = random.randint(0, len(wave.e)-1)
        if coin > 0.5:
            wave.e = np.delete(wave.e, index)
        else:
            wave.e[index] = wave.e[index] * (1 - random_sign()*0.5)

        return wave

    def eval(self, x):
        tmp = np.zeros((x.shape[0]), dtype=np.complex64)

        for k, ck in enumerate(self.e):
            tmp += ck * np.exp(2j * np.pi * k)
            if k != 0:
                tmp += ck.conjugate() * np.exp(-2j * np.pi * k)

        return tmp.real

    def fitness(self, real_sol):
        d = decode_wave(self)

        if len(d) != len(real_sol):
            return -1

        if not np.equal(real_sol, d).all():
            return -1

        self.strength = len(self.e)

        return self.strength




'''
n = 7
data = np.reshape(sols, (-1,)).tolist()
data.insert(0, data[-1])
x, y = build_fun(sols[0])
e = np.fft.rfft(y)
pop = 1000
iterations = 500
pops = []
for _ in range(pop):
    pops.append(Wave(n, e))




for o in range(iterations):
    print(o, pops[0].strength)
    l = len(pops)
    for i in range(l):
        mutant = pops[i].mutate()
        if mutant.fitness(data[1:]) > 0:
            pops.append(mutant)

    pops.sort(key=lambda x: x.strength)
    pops = pops[:pop]

print(data[1:])
for p in range(10):
    print(pops[p].strength, decode_wave(pops[p]))


fig, ax = plt.subplots()
print(len(e), len(pops[0].e))
ax.scatter(e.real, e.imag, picker= 4)
ax.scatter(pops[0].e.real, pops[0].e.imag, picker= 5)
plt.show()


plt.plot(np.fft.irfft(pops[0].e))
plt.show()
'''


sols = nqueens.n_queens(n)
all_sols = sols.all_solutions
sorted_sols = [all_sols[0]]
for i in range(1, len(all_sols)):
    diff = []
    for sol in all_sols:
        dist = np.linalg.norm(np.array(sorted_sols[-1])-np.array(sol))
        if dist == 0 or sol in sorted_sols:
            dist = n*n
        diff.append(dist)
    sorted_sols.append(all_sols[np.argmin(diff)])



ee = []
x_all = []
for sol in sorted_sols:
    x, y = build_fun(sol)
    #c = fourier_series_coeff_numpy(x, return_complex=True)[1:49]
    #plt.plot(x+y)
    #plt.show()
    x_all.append(x[:49])
    #_, axf, bxf = fourier_series_coeff_numpy(x)
    #_, ayf, byf = fourier_series_coeff_numpy(y)
    #plt.plot(axf[:49], c='r')
    #plt.plot(bxf[:49], c='g')

    #plt.plot(ayf[:49], c='g')
    #plt.plot(byf[:49], c='g')
    #plt.show()
    #plt.plot(bf[1:49])


    #xx, yx = wave_from_fourier_coff(a0, af, bf)
    #plt.plot(x)
    #plt.plot(yx)
    #plt.show()
    #ee.append(axf[:49])

x_all = np.array(x_all)
f2d = np.fft.rfft2(x_all)
#plt.plot(f2d[:,1])
print(f2d.shape)
#plt.show()

#ee = np.array(ee)

for i in range(15):
    #ax.scatter(ee[:,i].real, ee[:,i].imag)
    plt.plot(f2d[i])
plt.show()



# how to measure "uniformidad", its enough with entropy?
# Create random vector of radious (at least a diff of 0.1) [0.1, 1.0]
# Concatenate random numbers of solutions.
# Sum random solutions, maybe there is a pattern there to separate them later.
# Create random vector of radious  [all separated by 1] [1, N] (must be the same or equivalent than before)