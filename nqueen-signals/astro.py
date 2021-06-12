import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import entropy
import nqueens
import itertools
import random
import remix
import matplotlib.animation as animation
import scipy.stats as stats

steps = 100
theta = np.linspace(0, 2*np.pi, steps)


def get_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    return m, b


def eval_equation(m, b, x):
    return x, m*x + b

def convert_list(data, n, set_size):
    data = data.astype('uint8')
    num_bits = math.floor(math.log2(n-1)) + 1

    if num_bits*len(data) % set_size != 0:
        return np.array([])

    return np.reshape(np.packbits(np.flip(np.reshape(np.unpackbits(np.reshape(data, (-1, 1)), axis=1)[:, -num_bits:], (-1, set_size)), axis=1), bitorder='little', axis=-1), -1)


orbit_count = 0
def orbit(p1, p2, pie=1, N=100):
    global orbit_count
    x2, y2 = p2
    angle_init = orbit_count*2*np.pi/pie
    angle_end = orbit_count*2*np.pi/pie + 2*np.pi/pie
    orbit_count += 1

    theta = np.linspace(angle_init, angle_end, N)
    x = (x2+1)*np.cos(theta) - x2 + 1
    y = (x2+1)*np.sin(theta)

    return x, y


def series_complex_coeff(c, t, T):
    tmp = np.zeros((t.size), dtype=np.complex64)
    for k, ck in enumerate(c):
        # sum from 0 to +N
        tmp += ck * np.exp(2j * np.pi * k * t / T)
        # sum from -N to -1
        if k != 0:
            tmp += ck.conjugate() * np.exp(-2j * np.pi * k * t / T)
    return tmp.real


def build_fun(sol, pie):
    y = []
    for i in range(len(sol)-1):
        s1 = sol[i]
        s2 = sol[i+1]
        xx, yy = orbit((s1, np.sin(pie)), (s2, np.sin(pie)), pie)
        y.extend(yy)

    x = np.linspace(0, np.pi*len(sol), len(y))

    #fig, ax = plt.subplots()
    #ax.plot(x, y)
    #plt.show()

    return x, y


def build_random_fun(sol):
    y = []
    for i in range(len(sol)-1):
        s1 = sol[i] + random.random() - 0.5
        s2 = sol[i+1] + random.random() - 0.5
        xx, yy = orbit((s1, 0), (s2, 0))
        y.extend(yy)

    x = np.linspace(0, 2*np.pi*len(sol), len(y))

    return x, y

def decode_wave(wave, pie, max_num_bits, n, N=100):
    wave_round = np.abs(np.round(np.array(wave))[N//2::N]) # *math.pow(2, max_num_bits)
    num_bits = math.floor(math.log2(n-1)) + 1
    return convert_list(wave_round, math.pow(2, max_num_bits), num_bits)


def check_fourier_coff(pie, max_num_bits):
    n = 6
    sols = nqueens.n_queens(n).all_solutions

    all_perm = itertools.permutations(sols)

    for i, sol in enumerate(all_perm):
        if i in [1]: #, 9, 11, 12]:
            oo = np.reshape(sol, (-1,))
            wave = convert_list(np.reshape(sol, (-1,)), n, max_num_bits)
            wave = np.insert(wave, 0, wave[-1])
            wave = wave #/math.pow(2, max_num_bits+1)
            #print(wave)

            x, y = build_fun(wave, pie)

            dd = decode_wave(y, pie, max_num_bits, n)
            #print(np.all(oo == dd))

            coff = np.fft.rfft(y)
            fig, ax = plt.subplots()
            lim = 100
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.scatter(coff.real, coff.imag, picker=5)
            plt.show()
            break


#check_fourier_coff(4, 6)


def super_checker(n):
    sols = nqueens.n_queens(n).all_solutions

    all_perm = itertools.permutations(sols)

    for i, sol in enumerate(all_perm):
            nsol = np.reshape(sol, (-1,))
            for max_num_bits in range(2, 8):
                for pie in range(1, 8):
                    #max_num_bits = random.randint(2, 8)
                    #pie = random.randint(1, 8)
                    wave = convert_list(nsol, n, max_num_bits).tolist()
                    if len(wave) == 0:
                        continue
                    wave = np.insert(wave, 0, wave[-1])
                    wave = wave #/math.pow(2, max_num_bits)

                    x, y = build_fun(wave, pie)

                    #dd = decode_wave(wave, pie, max_num_bits, n)

                    coff = np.fft.rfft(y)
                    fig, ax = plt.subplots()
                    lim = 100
                    ax.set_xlim(-lim, lim)
                    ax.set_ylim(-lim, lim)
                    ax.scatter(coff.real, coff.imag, picker=5)
                    plt.show()


#super_checker(6)


def super_checker6(n, i, max_num_bits, pie):

    sols = nqueens.n_queens(n).all_solutions

    all_perm = itertools.permutations(sols)
    for _ in range(i):
        next(all_perm)

    nsol = np.reshape(next(all_perm), (-1,))
    wave = convert_list(nsol, n, max_num_bits).tolist()
    wave = np.insert(wave, 0, wave[-1])

    x, y = build_fun(wave, pie)

    #dd = decode_wave(wave, pie, max_num_bits, n)

    coff = np.fft.rfft(y)
    fig, ax = plt.subplots()
    lim = 100
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    crside = coff.real[(coff.real >= -15) & (coff.imag >= -50) & ~((coff.imag < -0.1) & (coff.real < 12)) & ((coff.real < 100) & (coff.imag < 100) & (coff.imag > -100) )]
    ciside = coff.imag[(coff.real >= -15) & (coff.imag >= -50) & ~((coff.imag < -0.1) & (coff.real < 12)) & ((coff.real < 100) & (coff.imag < 100) & (coff.imag > -100) )]

    p11, p12 = (-9.071401330982098, 0.0), (-12.62860400116181, 71.64701226474702)
    p21, p22 = (-11.394672835880272, 16.31155884526794), (91.79719485161468, 94.18895216963975)
    p31, p32 = (11.817107745948988, 15.32446988172049), (91.07674908774428, -42.54097619512062)
    p41, p42 = (-2.045191798168375, 21.315973607858318), (88.26411980441915, 32.019679337667355)
    p51, p52 = (-0.6240301122188612, 21.022441205163236), (86.4435589015708, 25.30211106908183)
    p61, p62 = (2.343037843840648, 21.13503601089934), (90.39363434180169, 12.723029534741068)

    ax.plot([p11[0], p12[0]], [p11[1], p12[1]])
    ax.plot([p21[0], p22[0]], [p21[1], p22[1]])
    ax.plot([p31[0], p32[0]], [p31[1], p32[1]])
    ax.plot([p41[0], p42[0]], [p41[1], p42[1]])
    ax.plot([p51[0], p52[0]], [p51[1], p52[1]])
    ax.plot([p61[0], p62[0]], [p61[1], p62[1]])

    eqps = [(p11, p12), (p21, p22), (p31, p32), (p41, p42), (p51, p52), (p61, p62)]

    l1 = [[-9.74182996116598, 31.05445796679838], [-9.432264959970762, 22.779513974210815], [-9.373463121879805, 20.840425741146987], [-9.289259924324444, 17.698061162635156]]
    l2 = [[-9.8871213249666, 17.449285922053942], [-9.377577495069474, 17.833831198209683], [-6.168376980937843, 20.255767890476218],
          [-5.596187940573113, 20.687590590761662], [-5.009353274893755, 21.13046613027565], [-4.406097575774979, 21.585734383562816],
          [-3.784477783306098, 22.05486174951618], [-1.067948278517001, 24.1049868365295], [-10.39246364514969, 17.067911463831116]]

    l4 = [[-2.045191798168375, 21.315973607858318], [-1.3764184738060692, 21.39470319071813], [-0.7054154152896785, 21.473704021605784],
    [-0.03083834210281111, 21.553134465430254], [2.7313393484722113, 21.878468668136392]]

    l6 = [[3.676933085622636, 21.00935766550323], [2.343037843840648, 21.13503601089934]]

    lall = []
    lall.extend(l1)
    lall.extend(l2)
    lall.extend(l4)
    lall.extend(l6)


    xrs = [np.array(l1)[:,0].tolist(), np.array(l2)[:,0].tolist(), [], np.array(l4)[:,0].tolist(), [], np.array(l6)[:,0].tolist()]
    yrs = [np.array(l1)[:,1].tolist(), np.array(l2)[:,1].tolist(), [], np.array(l4)[:,1].tolist(), [], np.array(l6)[:,1].tolist()]

    # dist = np.linalg.norm(a-b)

    for r, i in zip(crside, ciside):
        ci = 0
        dist_min = math.inf
        if [r, i] in lall:
            continue
        for ieq, eq in enumerate(eqps):
            dist = np.abs(np.linalg.norm(np.cross(np.array(eq[1])-np.array(eq[0]), np.array(eq[0])-np.array([r,i])))/np.linalg.norm(np.array(eq[1])-np.array(eq[0])))
            if dist < dist_min:
                dist_min = dist
                ci = ieq
        xrs[ci].append(r)
        yrs[ci].append(i)

    for i in range(len(xrs)):
        xo = np.array(xrs[i])
        yo = np.array(yrs[i])
        xrs[i] = xo[np.argsort(xo)].tolist()
        yrs[i] = yo[np.argsort(xo)].tolist()

    for xx, yy in zip(xrs, yrs):
        ax.scatter(xx, yy, picker=2)
        d = []
        #po = np.array(list(zip(xx, yy)))
        for i in range(len(xx)-1):
            d.append(np.linalg.norm(np.array([xx[i+1], yy[i+1]])-np.array([xx[i], yy[i]])))
        r = []
        for i in range(len(d)-1):
            r.append(d[i+1]/d[i])
        plt.plot(list(range(len(r))), r)


    #crplus = coff.real[coff.real >= 0]
    #ciplus = coff.imag[coff.real >= 0]

    #crminus = coff.real[coff.real < 0]
    #ciminus = coff.imag[coff.real < 0]

    #ax.scatter(coff.real, coff.imag, picker=5)
    #ax.scatter(crplus, ciplus, picker=5)
    #ax.scatter(crminus, ciminus, picker=2)
    #ax.scatter(crside, ciside, picker=2)

    def onpick3(event):
        index = event.ind
        xy = event.artist.get_offsets()
        #print(xy[index])

    fig.canvas.callbacks.connect('pick_event', onpick3)

    plt.show()


#super_checker6(6, 1, 6, 4)

# 0 3 1
# 0 3 4
# 0 6 1
# 0 6 2
# 0 6 3
# 0 6 4
# 1 3 1
# 1 3 2
# 1 3 3
# 1 4 4
# 1 6 4
# 2 3 1
# 2 3 2
# 2 6 2
# 2 6 4
# 3 3 1
# 3 3 2
# 3 6 1
# 3 6 2
# 3 6 4


def super_checker(all_perm, i, max_num_bits, pie):

    for _ in range(i):
        next(all_perm)

    nsol = np.reshape(remix.mix_vertical(next(all_perm)), (-1,))
    wave = convert_list(nsol, n, max_num_bits).tolist()
    wave = np.insert(wave, 0, wave[-1])

    x, y = build_fun(wave, pie)


    coff = np.fft.rfft(y)
    ll = 25
    #coff1imag = coff[(coff.imag < ll) & (coff.imag > -ll)]
    #coff1ireal = coff[(coff.real < ll) & (coff.real > -ll)]
    #print('len coff', len(coff), len(coff1imag))


    fig, ax = plt.subplots()
    lim = 10000
    ax.plot(coff.real)
    #ax.set_xlim(-ll, ll)
    #ax.set_ylim(0, 700)

    #ax.scatter(coff.real, coff.imag, picker=5)
    #ax.hist(coff.real, bins=30)
    #ax.hist(coff1ireal.real, bins=100)

    #plt.plot(coff_coffreal.real)


    coff_1 = np.roll(coff, -1)
    coff_op = (coff_1 * coff)[:-1]

    '''
    line, = ax.plot(np.array([]), np.array([]), picker=5)
    #coff_op = coff_op[(coff_op.real < 5000) & (coff_op.real > -5000)]

    def animate(i):
        print(i)
        line.set_xdata(coff_op.real[300 + i-40:i + 300])
        line.set_ydata(coff_op.imag[300 + i-40:i+ 300])
        return line,

    ani = animation.FuncAnimation(fig, animate, interval=500, blit=True, save_count=50)
    '''

    '''
    coff3 = coff_op[::9]
    coff3_mid = (coff3[::2]+coff3[1::2])/2
    ax.scatter(coff3_mid.real, coff3_mid.imag, picker=5)

    coff3_midp = coff3_mid[(coff3_mid.real < 500) & (coff3_mid.real > -500)]
    l = []
    for r, i in zip(coff3_mid.real, coff3_mid.imag):
        l.append((r - coff3_mid.real[-1], i - coff3_mid.imag[-1]))
    '''



    '''
    for i in range(9):
        coffs = coff_op[i::9]
        ax.scatter(coffs.real, coffs.imag, picker=5)
    '''


    #ax.plot(coff_op.real, coff_op.imag, picker=5)
    plt.show()


    return coff_op

'''
n = 6
sols = nqueens.n_queens(n).all_solutions
v = itertools.permutations(list(range(len(sols))))
all_perm = itertools.permutations(sols)
coff_op = super_checker(all_perm, 1, n, 1)
'''

# 1500 6000 14000 140000 300000
#30 120 280 2800 6000

# 8 => 50
# 9 => 49
# 10 => 50

n = 6
sols = nqueens.n_queens(n).all_solutions
v = itertools.permutations(list(range(len(sols))))
all_perm = itertools.permutations(sols)
for _ in range(100):
    coff_op = super_checker(all_perm, 1, 4, 2)

