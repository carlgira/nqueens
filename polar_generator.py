import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import nqueens
import numpy as np
import math
from itertools import permutations
import itertools
import random


# Setup


#perm = permutations(list(range(n)))
#all_permutations = [list(p) for p in list(perm)]






'''
def validate(sol):
    if len(sol) != len(set(sol)):
        return False
    v1 = [sol[i] + i for i in range(len(sol))]
    if len(v1) != len(set(v1)):
        return False
    v2 = [sol[i] - i for i in range(len(sol))]
    if len(v2) != len(set(v2)):
        return False
    return True


            
'''



import numpy as np
from scipy import fftpack
import pylab as pl




'''
import numpy as np
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from sklearn.model_selection import train_test_split

a_values = []

sols = []
non_sols = []

for sol in all_permutations:
    sig = np.array([queenWave(x, sol) for x in time_vec])
    sig_fft = fftpack.fft(sig)
    coeff = [math.sqrt(cn.real**2 + cn.imag**2) for cn in sig_fft]
    coeff = (coeff-np.min(coeff))/(np.max(coeff)-np.min(coeff))
    if validate(sol):
        sols.append(coeff[0:n*n])
        for _ in range(100):
            x_data.append(coeff[0:n*n])
            y_data.append([1.0, 0.0])
    else:
        non_sols.append(coeff[0:n*n])
        x_data.append(coeff[0:n*n])
        y_data.append([0.0, 1.0])




batch_size = 100
learning_rate = 0.0001
training_epochs = 12
num_classes = 2

combined = list(zip(x_data, y_data))
random.shuffle(combined)
x_data[:], y_data[:] = zip(*combined)

x_data = np.reshape(x_data, (-1, n*n))
y_data = np.reshape(y_data, (-1, num_classes))

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

model = Sequential()
model.add(Dense(n*n*2, activation='relu'))
model.add(Dense(n, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=training_epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(x_data, y_data, verbose=0)

print(np.shape(x_data), np.shape(y_data))

print('All Test accuracy:', score[1])

x_data = np.reshape(sols, (-1, n*n))
y_data = np.reshape([[1.0, 0.0]]*len(sols), (-1, num_classes))
score = model.evaluate(x_data, y_data, verbose=0)
print('Sols Test accuracy:', score[1])

x_data = np.reshape(non_sols, (-1, n*n))
y_data = np.reshape([[0.0, 1.0]]*len(non_sols), (-1, num_classes))
score = model.evaluate(x_data, y_data, verbose=0)
print('NSols Test accuracy:', score[1])


main_sig = fftpack.ifft(sig_fft)
#print(main_sig)

pl.figure()
pl.plot(time_vec, sig, c='b', linewidth=3)
pl.plot(time_vec, main_sig, c='r')
#pl.plot(time_vec, sig_fft, c='y')
#pl.plot(time_vec, coeff, c='g')
pl.xlabel('Time [s]')
pl.ylabel('Amplitude')
#plt.show()

an = coeff[0:int(len(coeff)/2)]
an = np.reshape(an, (n,n))
print(an)
for p in range(n):
    pl.plot(an[:, p])

#plt.show()
'''

'''
n = 7
from super_queens import SuperQueens

phi = (1 + math.sqrt(5))/2
c2 = math.pow(phi, 1/(3*math.pi))

#lsols = get_sol(c2)
sq = SuperQueens(n)
lsols = sq.get_sols(c2)

lsetsols = list(set([tuple(sol) for sol in lsols]))
print(len(lsols), len(lsetsols))

lsetsols = [ validate(sol) for sol in lsols]


print(lsetsols)

time_step = 0.02
r = calc_r(n)


x_data = []
y_data = []


x_ = np.linspace(-0.5*math.pi/n, (n*2-0.5)*math.pi/n, n*n*n*2+2)
time_vec = x_[1:-1]

y = 20*np.cos(n*time_vec)

sols = nqueens.n_queens(n).group_solutions()[4]

print(sols)

sum_sols = np.zeros(len(time_vec))


np.zeros(len(time_vec))
mask = None

for sol in lsols:
    sig = np.array([queenWave(x, sol) for x in time_vec])
    sig = sig[sig != np.array(None)]
    sig_fft = fftpack.fft(sig)
    isig = fftpack.ifft(sig_fft)
    sum_sols = sum_sols + isig


def check_formula():

    result = r[0]*np.cos(time_vec)
    for k in range(1, n*n):
        result += r[k % n]*np.cos(time_vec*(k+1))
    return result

yy = check_formula()

pl.figure()
#pl.plot(time_vec, sum_sols, c='b', linewidth=3)
#pl.plot(time_vec, y, c='r', linewidth=1)
pl.plot(time_vec, yy, c='g', linewidth=1)
pl.plot(time_vec, 0.1*np.cos(n*time_vec), c='y', linewidth=1)
pl.plot(time_vec, [2.1]*len(time_vec), c='r', linewidth=1)
pl.plot(time_vec, [1.7]*len(time_vec), c='b', linewidth=1)
pl.xlabel('Time [s]')
pl.ylabel('Amplitude')
plt.show()

'''

cosmx_cosnx = lambda x, m, n: math.sin(x*(m-n))/(2*(m-n)) + math.sin(x*(m+n))/(2*(m+n))
cosmx2 = lambda x, m: m*x/4 + math.sin(2*m*x)/8
cosmx_sinnx = lambda x, m, n: 0.5*(math.cos(x*(m-n))/(m-n) - (math.cos(x*(m+n)))/(m+n))

cosmx_sinmx = lambda x, m: -(math.cos(x)**2)/(m*2)

class QueenFourier:

    def __init__(self, n, iterations, sol=None):
        self.n = n
        self.iterations = iterations
        self.r = self.calc_r()
        self.ans = [(-0.5 + i)*math.pi/self.n for i in range(0, (self.n+1)*2, 2)]
        self.a0 = 0
        self.an = []
        self.bn = []
        self.anlm = []
        self.bnlm = []
        self.cs = self.r
        if sol is not None:
            self.cs = [self.r[i] for i in sol]

    def calc_r(self):
        phi = (1 + math.sqrt(5))/2
        c = pow(phi, 2/math.pi)
        angle = math.pi*2/self.n
        a = [angle*(i+1) for i in range(self.n)]
        spiral_func = lambda t, l: math.pow(c, t - 12*math.pi + l)
        r = []
        r.append(spiral_func(8*math.pi + angle, 2*math.pi))
        for aa in a[:-1]:
            r.append(spiral_func(10*math.pi + angle, aa))
        return r

    def calc_a0(self):
        result = 0
        anl = []
        for i in range(len(self.ans)-1):
            value = ((math.sin(self.ans[i+1]*self.n) - math.sin(self.ans[i]*self.n))/self.n)/(2*math.pi)
            anl.append(value)
            result += value*self.cs[i]
        self.anlm.append(anl)
        return result

    def calc_an(self, m):
        result = 0
        anl = []
        for i in range(len(self.ans)-1):
            value = (cosmx_cosnx(self.ans[i+1], self.n, m) - cosmx_cosnx(self.ans[i], self.n, m))/(math.pi)
            anl.append(value)
            result += value*self.cs[i]
        self.anlm.append(anl)
        return result

    def calc_ak(self):
        result = 0
        anl = []
        for i in range(len(self.ans)-1):
            value = (cosmx2(self.ans[i+1], self.n) - cosmx2(self.ans[i], self.n))/(2*math.pi)
            anl.append(value)
            result += value*self.cs[i]
        self.anlm.append(anl)
        return result

    def calc_bk(self):
        result = 0
        bnl = []
        for i in range(len(self.ans)-1):
            value = (cosmx_sinmx(self.ans[i+1], self.n) - cosmx_sinmx(self.ans[i], self.n))*self.r[i]/(math.pi)
            bnl.append(value)
            result += value*self.cs[i]
        self.bnlm.append(bnl)
        return result

    def calc_bn(self, m):
        result = 0
        bnl = []
        for i in range(len(self.ans)-1):
            value = (cosmx_sinnx(self.ans[i+1], self.n, m) - cosmx_sinnx(self.ans[i], self.n, m))/(math.pi)
            bnl.append(value)
            result += value*self.cs[i]
        self.bnlm.append(bnl)
        return result

    def calc_coeff(self):
        self.a0 = self.calc_a0()
        for m in range(1, self.iterations):
            if m == self.n:
                pass
                self.an.append(self.calc_ak())
                self.bn.append(self.calc_bk())
            else:
                self.an.append(self.calc_an(m))
                self.bn.append(self.calc_bn(m))

    def eval(self, x):
        y = np.array([self.a0]*(len(x)))
        #y = np.zeros(x.shape)
        for i in range(len(self.an)):
            y += self.an[i]*np.cos(x*i)
            y += self.bn[i]*np.sin(x*i)
        return y

    def queen_wave(self, x):
        for i in range(self.n):
            if (-0.5 + 2*i)*math.pi/self.n < x < (1.5 + 2*i)*math.pi/self.n:
                return self.cs[i]*math.cos(x*self.n)



def validate(sol):
    if len(sol) != len(set(sol)):
        return False

    v1 = [sol[i] + i for i in range(len(sol))]

    if len(v1) != len(set(v1)):
        return False

    v2 = [sol[i] - i for i in range(len(sol))]

    if len(v2) != len(set(v2)):
        return False

    return True


from super_queens import SuperQueens


def get_v(l):
    for n in l:

        qq = QueenFourier(n, 10)
        su = SuperQueens(n)
        init_sols = [m.tolist() for m in su.init_sols()]
        stuff = qq.r
        count = 0
        sums = []
        all = []

        nsols = nqueens.n_queens(n).all_solutions

        x = np.linspace(0, math.pi, 100)
        pl.figure()

        values = []

        ree = {}
        #nsols = itertools.permutations(list(range(n)), n)
        x_all = []
        y_all = []
        for sol in nsols:
            y = 0
            for i, l in enumerate(sol):
                y += stuff[l]*np.cos(x*(i+1))
            v = round(y[-1], 8)


            pl.plot(x, y, c='r')

            x_all.append(x)
            y_all.append(y)

            if v not in ree:
                ree[v] = []
            ree[v].append(sol)

            values.append(v)




        values.sort()
        print(n, len(values), len(set(values)))
        set_values = list(set(values))
        set_values.sort()
        print(set_values)
        print(ree)

        cc = {}
        for k in ree:
            for l in ree[k]:
                if l in init_sols:
                    if k not in cc:
                        cc[k] = 0
                    else:
                        cc[k] = 1

        #print(len(cc))
        #print(cc)


    pl.show()

import copy

def get_z_index(n):

    class Pop:
        def __init__(self, n, x, nsols):
            if n is not None:
                self.nsols = nsols
                self.z = self.gen_z(n)
                self.obj = self.create_obj(x, self.z)
                self.pos, self.score =  self.get_best_x(self.obj)

        def create_obj(self, x, ll):
            y_all = []
            for index, sol in enumerate(self.nsols):
                y = 0
                for i, l in enumerate(sol):
                    y += stuff[l]*np.cos(x*(i+1) + ll[index])
                y_all.append(y)

            return np.reshape(y_all, (-1, len(x)))

        def gen_z(self, n):
            return [math.pi*random.random() for _ in range(len(self.nsols))]

        def get_best_x(self, y_all):
            m = np.linalg.norm( y_all - np.mean(y_all, axis=0), axis= 0)
            return np.argmin(m), np.min(m)

        def mutate(self, x):
            cc = Pop(None, None, None)
            cc.nsols = self.nsols
            ci = random.randint(0, n-1)
            cc.z = [v for v in self.z]
            cc.z[ci] = cc.z[ci] + [-1,1][random.randrange(2)]*(random.random()*0.01)
            cc.obj =  cc.create_obj(x, cc.z)
            cc.pos, cc.score =  cc.get_best_x(cc.obj)
            return cc

        def set_z(self, nsols, z):
            self.nsols = nsols
            self.z = z
            self.obj = self.create_obj(x, self.z)
            self.pos, self.score =  self.get_best_x(self.obj)

    population = 1000
    iterations = 1000
    x = np.linspace(1, math.pi, 100)


    qq = QueenFourier(n, 10)
    stuff = qq.r
    count = 0

    nsols = nqueens.n_queens(n).all_solutions

    pops = [Pop(n, x, nsols) for _ in range(population)]
    bb = [2.1068185686837775, -0.022686051979521588, 2.5124076171581438, 1.8211638706128268, 0.8515882595577762, 2.604482943743224, 0.5113499970646882, 0.3461167054681471, 2.6428372026159312, 3.0929012854948077]
    for _ in range(population):
        g = Pop(None, None, None)
        g.set_z(nsols, bb)
        pops.append(g)
    pops.sort(key=lambda x: x.score)

    for it in range(iterations):
        print(it)
        for p in range(population):
            if random.random() < 0.5:
                nobj = pops[p].mutate(x)
                pops.append(nobj)

        pops.sort(key=lambda x: x.score)
        pops = pops[:population]


    for p in pops:
        print(p.score)

    o = pops[0]
    print(o.pos, x[o.pos], o.score)
    print(o.z)
    for isol in range(len(o.nsols)):
        pl.plot(x, o.obj[isol], c='r')
    pl.show()

# [2.1028141201789143, -0.02139215529338709, 2.5124076171581438, 1.8197244587758086, 0.8477116778353716, 2.604482943743224, 0.5113499970646882, 0.3461167054681471, 2.6428372026159312, 3.0929012854948077]
# [2.1112458017508167, -0.026012300317413, 2.5124076171581438, 1.8252219790687816, 0.8477116778353716, 2.604482943743224, 0.5113499970646882, 0.3461167054681471, 2.6428372026159312, 3.0929012854948077]


# 30 1.6489674707847857 0.08318279485761841
#[2.106859578475762, -0.022432846371169293, 2.512038607255944, 1.8206264815169102, 0.8519442735889581, 2.604482943743224, 0.5113499970646882, 0.3461167054681471, 2.6428372026159312, 3.09290
get_z_index(5)









import scipy
import scipy.integrate as integrate

def build_f(a, sol):
    return lambda x: sum([a[l]*math.cos(x*(i+1)) for i, l in enumerate(sol)])



def get_integration_values(n):

    print('fff')
    qq = QueenFourier(n, 10)
    a = qq.r
    print('fff1')

    nsols = itertools.permutations(list(range(n)), n)
    int_sol = []
    int_nsol = []

    for sol in nsols:
        if validate(sol):
            int_sol.append(scipy.real(integrate.quad(  build_f(a, sol), 0, math.pi/2))[0])
        else:
            int_nsol.append(scipy.real(integrate.quad(  build_f(a, sol), 0, math.pi/2))[0])

    #print('sols', int_sol)

    #print('nsols', int_nsol)
    return int_sol, int_nsol


def laplace_transform():
    n = 8
    qq = QueenFourier(n, 10)
    stuff = qq.r

    def calc(x, sol):
        return np.sum([(stuff[c]*x)/(np.power(x, 2) + (i+1)**2) for i, c in enumerate(sol)], axis=0)

    #for sol in nqueens.n_queens(7).all_solutions:

    '''
    for sol in itertools.permutations(list(range(n)), n):
        x = np.linspace(0.5, 4.5, 100)
        y = calc(x, sol)
        if validate(sol):
            pass
        else:
            plt.plot(x,y, c='b')
    '''

    #for sol in nqueens.n_queens(7).all_solutions:
    for sol in itertools.permutations(list(range(n)), n):
        x = np.linspace(0.5, 4.5, 100)
        y = calc(x, sol)
        if validate(sol):
            plt.plot(x,y, c='r')

    plt.savefig('8all.png')



def count_sols_pos(n):
    sols = nqueens.n_queens(n).all_solutions
    vars = list('abcdefghijklm'[0:n])
    d = {}
    for v in vars:
        d[v] = 0

    result = [d.copy() for _ in range(n)]

    for sol in sols:
        for i, v in enumerate(sol):
            result[i][vars[v]] = result[i][vars[v]] + 1

    one = d.copy()
    two = d.copy()
    three = []

    for i, v in enumerate(result):
        three.append(sum(list(v.values())))
        if i % 2 == 0:
            for k in v.keys():
                one[k] = one[k] + v[k]
        else:
            for k in v.keys():
                two[k] = two[k] + v[k]

    return result, one, two, three

'''
cs, one, two, three = count_sols_pos(7)
print (cs)

print (one, two, three)

for v in cs:
    print(list(v.values()))
'''










