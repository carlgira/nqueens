import random
import numpy as np
import matplotlib.pyplot as plt
import math

mutate_rate = 0.1
phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l
c1 = math.pow(phi, 2/(math.pi))

def random_sign():
    return 1 if random.random() < 0.5 else -1

def get_points(n):
    t = 2/n
    if n % 2 == 0:
        return [round((-t/2 - t*((n-1)//2)) + i*t, 5) for i in range(n)]
    else:
        return [round((-t*(n//2)) + i*t, 5) for i in range(n)]


sols = [[2, 5, 1, 4, 0, 3], [1, 3, 5, 0, 2, 4], [3, 0, 4, 1, 5, 2], [4, 2, 0, 5, 3, 1]]
s1 = [2,1,3,4]


class Wave:
    def __init__(self, p, n):
        self.n = n
        self.p = p
        self.a0 = 0
        self.a = []
        self.b = []
        self.total = None

    def generate(self):
        self.a0 = round(random.random(), 3)
        self.a = np.array([round(random.random(), 3) for _ in range(self.p)])
        self.b = np.array([round(random.random(), 3) for _ in range(self.p)])

    def clone(self):
        wave = Wave(self.p)
        wave.n = self.n
        wave.a0 = self.a0
        wave.a = self.a[:]
        wave.b = self.b[:]
        return wave

    def mutate(self):
        wave = self.clone()
        coin = random.random()
        pos = random.randint(0, self.p-1)
        v = 1/(2*self.p + 1)
        if coin < v:
            wave.a0 = wave.a0 + random_sign() * wave.a0 * mutate_rate
        elif coin < (1-v)/2 + v:
            wave.a[pos] = wave.a[pos] + random_sign() * wave.a[pos] * mutate_rate
        else:
            wave.b[pos] = wave.b[pos] + random_sign() * wave.b[pos] * mutate_rate
        return wave

    def crossover(self, other):
        wave = self.clone()
        if random.random() < 0.5:
            wave.a[::2] = other.a[::2]
        else:
            wave.b[::2] = other.b[::2]
        return wave

    def eval(self, x):
        total = np.array([self.a0]*len(x))
        for i in range(0, len(self.a)):
            a = self.a[i]
            b = self.b[i]
            total += a*np.cos(x*(i+1)/self.n) + b*np.sin(x*(i+1)/self.n)
        self.total = total
        return self.total

    def plot(self, x):
        plt.plot(x, self.total)
        plt.show()

    def compare(self, points):
        pass

    def fitness(self):
        pass

n = 7
a = [(math.pi*2*i)/n for i in range(n)]
r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]






