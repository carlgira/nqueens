import itertools
import matplotlib.pyplot as plt
from collections import Counter
import nqueens
import numpy as np
from collections import Counter
from scipy import fftpack
import math

def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[x % base])
        x = x // base

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)

def get_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    return m, b

def data2binary(data, set_size=-1):

    num_bits = math.floor(math.log2(max(data))) + 1

    if num_bits*len(data) % set_size != 0:
        raise Exception('bad')

    if set_size == -1:
        set_size = num_bits

    l = np.flip(np.reshape(np.unpackbits(np.reshape(data, (-1, 1)), axis=1)[:, -num_bits:], (-1, set_size)), axis=1)
    return [''.join(v.tolist()) for v in l.astype(str)]

def binary2base(data, set_size=-1):
    return int(''.join(data2binary(data, set_size)), 2)


def factors(num):
    r = []
    temp = num
    i = 2
    while i < num+1:
        if temp % i == 0:
            r.append(i)
            temp = temp / i
            i -= 1
        i+= 1
    return r

# 40, 92, 352
# 46, 180, 1648,
# 44, 144, 1118

n = 8
x = np.linspace(0, np.pi, 100)
a = [(1/n)*i for i in range(1, n + 1)]
f = a[:]

sols = nqueens.n_queens(n)

def is_in_group(sol, sols):
    for i, g in enumerate(sols):
        if sol in g:
            return i
    return -1



def build_wave(sol, xx=x):
    y = np.zeros(len(xx))
    for i, v in enumerate(sol):
        yy = a[v]*np.cos((i+1)*xx)
        y = y + yy
    return xx, np.round(y, 5)

def build_wave_exp(sol, xx=x):
    cn = [a[v]/2 for i, v in enumerate(sol)] # Conver an, bn into cn
    y = np.zeros(len(xx))
    for i, c in enumerate(cn):
        y = y + c*np.exp(1j*(i+1)*xx)
        y = y + c*np.exp(-1j*(i+1)*xx)
    return xx, np.round(y, 5)

def all_graph(sols):
    for tt in sols:
        x, y = build_wave(tt)

        plt.plot(x, y)
        plt.show()

        #f_s = 100
        X = y
        freqs = fftpack.fftfreq(len(y)) #* f_sx

        fig, ax = plt.subplots()

        ax.stem(freqs, np.abs(X))
        ax.set_xlabel('Frequency in Hertz [Hz]')
        ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
        #ax.set_xlim(-f_s / 2, f_s / 2)
        #ax.set_ylim(-5, 110)
        plt.show()

        ffy = np.fft.irfft(y)
        plt.scatter(list(range(len(ffy))), ffy)
        plt.show()


last_y = []
marcs = []
o_groups = {}
waves = []
#xx = np.array([np.pi/9, 2*np.pi/9, np.pi/3, 4*np.pi/9, 5*np.pi/9, 2*np.pi/3, 7*np.pi/9, 8*np.pi/9, np.pi])
xx = np.array([np.pi/9, 2*np.pi/9, np.pi/4 ,np.pi/3, 4*np.pi/9, np.pi/2, 5*np.pi/9, 2*np.pi/3, 3*np.pi/4 , 7*np.pi/9, 6*np.pi/7 ,8*np.pi/9, np.pi])
for tt in sols.all_solutions:
    _, y = build_wave(tt, xx)
    waves.append(y)
    last_y.append(y[-1])
    marcs.append(y)

    if last_y[-1] not in o_groups:
        o_groups[last_y[-1]] = [tt]
    else:
        o_groups[last_y[-1]].append(tt)

c = Counter(last_y)
marcs = np.array(marcs)

waves = np.array(waves)
print("s", waves.shape)
wpos = []
for i in range(waves.shape[1]):
    wpos.append(len(set(waves[:,i].tolist())))

print(sorted(wpos))

for v in set(np.array(sorted(wpos, reverse=True)).tolist()):
    print(v, np.where(np.array(wpos) == v))


print('groups', len(sols.group_solutions()), len(sols.all_solutions))
print(len(c.keys()), c)
print(sorted(list(set(last_y))))

isols = itertools.permutations(list(range(n)), n)

gisols = []
count = 0
no_sols = []
yes_sols = []
o_sols = []

print('marcs', sorted(marcs[:, 8].tolist()))


'''
    if y[0] in marcs[:, 0].tolist() and \
            y[1] in marcs[:, 1].tolist() and \
            y[3] in marcs[:, 3].tolist() and \
            y[4] in marcs[:, 4].tolist() and \
            y[6] in marcs[:, 6].tolist() and \
            y[7] in marcs[:, 7].tolist() and \
            y[9] in marcs[:, 9].tolist() and \
            y[10] in marcs[:, 10].tolist() and \
            y[11] in marcs[:,11].tolist():
            
            
                if y[11] in marcs[:, 0].tolist() and \
            y[22] in marcs[:, 1].tolist() and \
            y[33] in marcs[:, 3].tolist() and \
            y[44] in marcs[:, 4].tolist() and \
            y[55] in marcs[:, 6].tolist() and \
            y[66] in marcs[:, 7].tolist() and \
            y[77] in marcs[:, 9].tolist() and \
            y[88] in marcs[:, 11].tolist() and \
            y[99] in marcs[:,12].tolist():
'''

for isol in isols:
    x, y = build_wave(isol, xx)


    if y[0] in marcs[:, 0].tolist() and \
            y[1] in marcs[:, 1].tolist() and \
            y[3] in marcs[:, 3].tolist() and \
            y[4] in marcs[:, 4].tolist() and \
            y[6] in marcs[:, 6].tolist() and \
            y[7] in marcs[:, 7].tolist() and \
            y[9] in marcs[:, 9].tolist() and \
            y[10] in marcs[:, 10].tolist() and \
            y[11] in marcs[:, 11].tolist() and \
            y[12] in marcs[:,12].tolist():

        # y[8] in marcs[:, 8].tolist() and \
        if nqueens.validate(isol):
            yes_sols.append(y[-1])
            #plt.plot(x, y, c='r')
        else:
            plt.plot(x, y, c='y')
            no_sols.append(y[-1])
            o_sols.append(isol)
        count += 1


gsols = sols.group_solutions()
print("count all", count)
print(o_sols)
yes_c = Counter(yes_sols)
no_c = Counter(no_sols)

print('yes', len(yes_c.keys()), yes_c)
print('no', len(no_c.keys()), no_c)


#def fss(x, Nh):
#    f = np.array([a[i]*np.exp(1j*2*i*x) for i in range(1,Nh+1)])
#    return f.sum()


for tt in sols.all_solutions:
    x, y = build_wave(tt)
    #y_Fourier_1 = np.array([fss(t,tt).real for t in x])
    yr = fftpack.fft(y)

    #yr = np.concatenate([yr[mtr-n:mtr], yr[mtr:mtr+n] ])

    #yr = np.append(yr[0], yr[3:-2])
    #xr, yy = build_wave_exp(tt)
    yy = fftpack.ifft(yr, 100)
    plt.plot(x, y, c='r')
    #plt.plot(x, yy, c='b')

    #ax.set_xlim(-f_s / 2, f_s / 2)
    #ax.set_ylim(-5, 110)

plt.show()

def get_freq(sol):
    x, y = build_wave(sol)

    X = fftpack.fft(y)

    freqs = fftpack.fftfreq(len(y))
    return freqs, X

def factors(num):
    r = []
    temp = num
    for i in range(2, num+1):
        if temp % i == 0:
            r.append(i)
            temp = temp / i
            i -= 1
    return r
