import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from fractions import Fraction
import math
import itertools
#from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp

phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l

def is_prime(n):
    return all(n%j for j in range(2, int(n**0.5)+1)) and n>1

def count_primes(n):
    count = 0
    for i in range(1, n+2, 2):
        if is_prime(i):
            print(i)
            count += 1
    return count

from math import sqrt

def divisors(n):
    divs = {1,n}
    for i in range(2,int(math.sqrt(n))+1):
        if n%i == 0:
            divs.update((i,n//i))
    return divs


def gen_symmetries(solution):

    symmetries = [solution]
    n = len(solution)

    x = list(range(n))
    for index in range(n):
        x[n - 1 - index] = solution[index]

    symmetries.append(x)

    #y(r[i]=j) → r[i]=n−j+1
    y = list(range(n))
    for index in range(n):
        y[index] = (n - 1 - solution[index])

    symmetries.append(y)

    #d1(r[i]=j) → r[j]=i
    d1 = list(range(n))
    for index in range(n):
        d1[solution[index]] = index

    symmetries.append(d1)

    # d2(r[i]=j) → r[n−j+1]=n−i+1
    d2 = list(range(n))
    for index in range(n):
        d2[n - 1 - solution[index]] = (n - 1 - index)

    symmetries.append(d2)

    # r90(r[i]=j) → r[j] = n−i+1
    r90 = list(range(n))
    for index in range(n):
        r90[solution[index]] = (n - 1 - index)

    symmetries.append(r90)

    # r180(r[i]=j) → r[n−i+1]=n−j+1
    r180 = list(range(n))
    for index in range(n):
        r180[n - 1 - index] = (n - 1 - solution[index])

    symmetries.append(r180)

    # r270(r[i]=j) → r[n−j+1]=i
    r270 = list(range(n))
    for index in range(n):
        r270[n - 1 - solution[index]] = index

    symmetries.append(r270)


    return symmetries


def clean_data(radious, theta, r, f=True):
    radious = np.array(radious)
    radious = radious[np.where(radious >= r)]
    if f:
        theta = theta[-len(radious):]
    else:
        theta = theta[:len(radious)]

    theta = theta[-len(radious):]
    return radious, theta


def draw(n, c1, c2, sol):

    f = [(2*i)/n for i in range(n)]

    a = [(math.pi*2*i)/n for i in range(n)]
    r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

    an = [(math.pi*2*i)/n for i in range(n)]
    rn = [spiral1(c1, 10*math.pi + math.pi/n, aa) for aa in an]

    a = np.array(a)
    r = np.array(r)

    an = np.array(an)
    rn = np.array(rn)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot2grid((1, 1), (0, 0), projection='polar')

    # Circles of principal
    for rr in r:
        theta = np.pi * np.arange(0, 2, 0.01)
        #ax.plot(theta, [rr]*len(theta))

    # Circles of sencondary
    for rr in rn:
        theta = np.pi * np.arange(0, 2, 0.01)
        #ax.plot(theta, [rr]*len(theta))

    for aa, ff in zip(a, f):
        theta1 = np.pi * np.arange(0, 12 - ff, 0.01)
        radious1 = [spiral1(c1, t, aa) for t in theta1]
        radious1, theta1 = clean_data(radious1, theta1, r[0])

        theta2 = np.pi * np.arange(2 - ff, 12, 0.01)
        radious2 = [spiral2(c1, t, aa) for t in theta2]
        radious2, theta2 = clean_data(radious2, theta2, r[0], False)

        #ax.plot(theta1, radious1, c='b')
        #ax.plot(theta2, radious2, c='r')

        ts1 = [spiral1_inv(c1, t, aa) for t in r]
        #ax.scatter(ts1, r, s=20, c='y')

        ts2 = [spiral1_inv(c1, t, aa) for t in rn]
        #ax.scatter(ts2, rn, s=20, c='g')

    theta3 = np.pi * np.arange(0, 12 - f[0], 0.01)
    radious3 = [spiral1(c2, t, a[0]) for t in theta3]
    radious3, theta3 = clean_data(radious3, theta3, r[0])

    theta4 = np.pi * np.arange(2 - f[0], 12, 0.01)
    radious4 = [spiral2(c2, t, a[0]) for t in theta4]
    radious4, theta4 = clean_data(radious4, theta4, r[0], False)

    ax.plot(theta3, radious3, c='black')
    ax.plot(theta4, radious4, c='gray')

    tt = [a[v] for v in sol]
    ax.scatter(tt, r, s=30, c='black')

    ax.set_rmax(1)
    plt.pause(0.001)
    plt.show()

#draw(7, math.pow(phi, 2/(math.pi)), math.pow(phi, 1/(2*math.pi)), [4, 1, 5, 2, 6, 3, 0])


class SuperQueens:

    def __init__(self, n):

        c1 = math.pow(phi, 2/math.pi)
        c = math.pow(phi, 2/math.pi)
        a = [(math.pi*2*i)/n for i in range(n)]
        r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

        self.a = np.array(a)
        self.r = np.array(r)
        self.solutions = []

        self.n = n
        self.c = c1

        self.mp = np.zeros((n, n, 2))

        for i in range(n):
            for j in range(n):
                self.mp[i][j] = [round(r[j]*math.cos(a[i]), 8), round(r[j]*math.sin(a[i]), 8)]

        self.mp_flatten = self.mp.reshape(-1, 2).tolist()

    def get_sol(self, c1, a=0):

        app = [spiral1_inv(c1, r, a) for r in self.r]
        ass = [spiral2_inv(c1, r, a) for r in self.r]

        xyp = [[round(r*math.cos(t), 8), round(r*math.sin(t), 8)] for r, t in zip(self.r, app)]
        xys = [[round(r*math.cos(t), 8), round(r*math.sin(t), 8)] for r, t in zip(self.r, ass)]

        s1 = []
        s2 = []

        for xysi, xypi in zip(xyp, xys):

            i_xysi = -1
            i_xypi = -1

            try:
                i_xysi = self.mp_flatten.index(xysi)
            except:
                pass

            try:
                i_xypi = self.mp_flatten.index(xypi)
            except:
                pass

            if i_xysi != -1:
                s1.append([i_xysi % self.n,int(i_xysi / self.n) ])

            if i_xypi != -1:
                s2.append([i_xypi % self.n, int(i_xypi / self.n)])

        return np.array(s1), np.array(s2)

    def validate(self, sol):

        if len(sol) != self.n:
            return False

        if len(sol) != len(set(sol)):
            return False

        v1 = [sol[i] + i for i in range(len(sol))]

        if len(v1) != len(set(v1)):
            return False

        v2 = [sol[i] - i for i in range(len(sol))]

        if len(v2) != len(set(v2)):
            return False

        return True

    def append_solutions(self, s):
        sols = gen_symmetries(s)
        for sol in sols:
            if self.validate(sol) and sol not in self.solutions:
                self.solutions.append(sol)

    @staticmethod
    def method_to_prove_spiral_values():

        for nn in range(5, 20):

            if not is_prime(nn):
                continue

            q = SuperQueens(nn)

            print(nn)
            rr = []
            rrv = []
            ii = 1
            for dd in range(1, nn+1):
                vv = Fraction(ii, dd)
                ii = vv.numerator
                dd = vv.denominator

                c = math.pow(phi, ii/(dd*math.pi))

                s1, s2 = q.get_sol(c)
                if tuple(s1[:,1].tolist()) not in rrv and (ii, dd) not in rr and q.validate(s1[:,1].tolist()) and q.validate(s2[:,1].tolist()):
                    rr.append((ii,dd))
                    rrv.append(tuple(s1[:,1].tolist()))
                    rrv.append(tuple(s2[:,1].tolist()))

            print(len(rr), rr, rrv)

    def all_filters(self):
        filters = []
        for i in range(2, int(self.n/2)+1):
            #values = [j<i for j in range(self.n)]
            #ll = list(set([a for a in itertools.permutations(values, len(values))]))
            ll = []
            for v in itertools.combinations(list(range(self.n)), i):
                t = np.array([False]*self.n)
                t[list(v)] = True
                ll.append(t.tolist())
            filters.append(np.array(ll))

        return filters


    def found_filters(self):

        max_c = int(self.n/2)
        sols = []

        all_filters = self.all_filters()

        for d in range(1, max_c):
            c = math.pow(phi, 1/(d*math.pi))
            for i, a in enumerate(self.a):
                s1, s2 = self.get_sol(c, a)
                sols.append([s1[:,1], 'f', d, i])
                sols.append([s2[:,1], 'b', d, i])


        sols_sym = []
        filter_formats = []
        sfilters = [[] for _ in range(int(self.n/2)+1)]

        for i in range(len(sols)-1):
            for j in range(i+1, len(sols)):
                sol1 = sols[i][0]
                sol2 = sols[j][0]
                filter = sol1 == sol2
                if np.sum(filter) == 1:
                    for f in all_filters:
                        for ff in f:

                            ffn = np.logical_not(ff)
                            r = ff*sol1 + ffn*sol2

                            if self.validate(r.tolist()) and not np.logical_and(filter, ff).any():
                                r_list = r.tolist()
                                print(np.sum(filter), sols[i], sols[j], ff, np.sum(ff), r.tolist(), np.where(sol1 == sol2)[0])
                                print(gen_symmetries(r.tolist()))

                                a_flag = False

                                sfilters[len(np.where(ff)[0].tolist())-1].append(np.where(ff)[0].tolist())

                                for k, syms in enumerate(sols_sym):
                                    if r_list in syms:
                                        a_flag = True
                                        filter_formats[k].append(
                                            {'d': [sols[i][1], sols[j][1]],
                                             'c': [sols[i][2], sols[j][2]],
                                             'a': [sols[i][3], sols[j][3]],
                                             'f': np.where(ff)[0].tolist(),
                                             'num.f': np.sum(ff),
                                             'q': np.where(filter)[0].tolist()[0],
                                             'qv': r[np.where(filter)].tolist()[0],
                                             'sol1': sol1.tolist(),
                                             'sol2': sol2.tolist()})
                                        break

                                if not a_flag:
                                    sols_sym.append(gen_symmetries(r.tolist()))
                                    filter_formats.append([{'d': [sols[i][1], sols[j][1]],
                                                            'c': [sols[i][2], sols[j][2]],
                                                            'a': [sols[i][3], sols[j][3]],
                                                            'f': np.where(ff)[0].tolist(),
                                                            'num.f': np.sum(ff),
                                                            'q': np.where(filter)[0].tolist()[0],
                                                            'qv': r[np.where(filter)].tolist()[0],
                                                            'sol1': sol1.tolist(),
                                                            'sol2': sol2.tolist()}])


        for i in range(len(sols_sym)):
            print(sols_sym[i])
            print()
            for j in sorted(set([tuple(v) for v in filter_formats[i]])):
                print(j)
            print()


        for i in range(len(sfilters)):
            print("i", i+1)
            print()
            for j in sorted(set([tuple(v) for v in sfilters[i]])):
                print(j)
            print()

    def init_sols(self):
        max_c = int(self.n/2)
        sols = []

        for d in range(1, max_c):
            c = math.pow(phi, 1/(d*math.pi))
            for i, a in enumerate(self.a):
                s1, s2 = self.get_sol(c, a)
                sols.append(s1[:,1])
                sols.append(s2[:,1])

        return sols

    def found_filters1(self, sols):
        all_filters = self.all_filters()

        for i in range(len(sols)-2):
            for j in range(i+1, len(sols)):
                for k in range(j+1, len(sols)):
                    sol1 = sols[i]
                    sol2 = sols[j]
                    sol3 = sols[k]
                    filter = sol1 == sol2
                    #if np.sum(filter) == 1:
                    for f in all_filters:
                        for ff in f:

                            ffn = np.logical_not(ff)
                            r = ff*sol1 + ffn*sol2

                            if self.validate(r.tolist()) and not np.isin(r, sols).all():
                                print(np.sum(filter), i, j, sols[i], sols[j], ff, np.sum(ff), r.tolist(), np.where(sol1 == sol2)[0])
                                print(gen_symmetries(r.tolist()))

    def get_sols(self, c1):
        sols = []
        for i, a in enumerate(self.a):
            s1, s2 = self.get_sol(c1, a)
            sols.append(s1[:,1].tolist())
            sols.append(s2[:,1].tolist())
        return sols

    def solve_eq(self):

        solver = pywrapcp.Solver("n-queens")


        #t1 = solver.IntVar(0, self.n-1, 't1')
        #t2 = solver.IntVar(0, self.n-1, 't2')

        tt = solver.IntVar(0, (self.n-1)*2, 't2')


        d = solver.IntVar(0, self.n-1, 'd')
        c = solver.IntVar(0, self.n-1, 'c')

        k = solver.IntVar(-self.n+1, self.n-1, 'k')

        q = [tt, c, d, k]

        #solver.Add(t1 != t2)
        #solver.Add(d != c)
        #solver.Add(d - c == k or c - d == k)
        #solver.Add( (t1 + t2) == 3)
        solver.Add((tt) == (7*self.n - (d+c) + k*self.n))

        db = solver.Phase(q, solver.CHOOSE_MIN_SIZE_LOWEST_MAX,solver.ASSIGN_CENTER_VALUE)

        solver.Solve(db)

        solver.NewSearch(db)

        sols = []
        sss = []
        while solver.NextSolution():
            r = [v.Value() for v in q]
            c = math.pow(phi, 1/(1*math.pi))
            s1, _ = self.get_sol(c, self.a[r[1]])
            _, s2 = self.get_sol(c, self.a[r[2]])

            filter = s1[:,1] == s2[:,1]
            if np.sum(filter) == 1:
                #print("ok", r, s1[:,1], s2[:,1 ])

                all_filters = self.all_filters()
                for f in all_filters:
                    for ff in f:

                        ffn = np.logical_not(ff)
                        rr = ff*s1[:,1] + ffn*s2[:,1 ]

                        if self.validate(rr.tolist()) and not np.logical_and(filter, ff).any():
                            print(r)
                            sss.append(r)
                            print("ok2", np.sum(filter), ff, np.sum(ff), rr.tolist(), np.where(s1[:,1 ] == s2[:,1 ])[0])
                            sols.append(tuple(rr.tolist()))

            else:
                print("ko", np.sum(filter), r, s1[:,1], s2[:,1 ])




        print(len(sols), len(set(sols)))
        sols = [list(s) for s in set(sols)]

        for i, s in enumerate(sols):
            sym = gen_symmetries(list(s))

            for e, ss in enumerate(sols):

                if i != e and ss in sym:
                    print(ss, i, e, "is sym")


        sss.sort()
        print("ss", sss)

    def new_sols(self):

        # f1
        c2 = math.pow(phi, 1/(1*math.pi))
        _ , s1 = self.get_sol(c2, self.a[int(self.n/2)])
        s2 , _ = self.get_sol(c2, self.a[(int(self.n/2)+2)])

        #1 [array([2, 0, 5, 3, 1, 6, 4]), 'b', 1, 3] [array([4, 6, 1, 3, 5, 0, 2]), 'f', 1, 5] [ True False False False False False  True] 2 [2, 6, 1, 3, 5, 0, 4] [3]

        ff = np.array([False]*self.n)
        ff[0] = True
        ff[-1] = True

        ffn = np.logical_not(ff)
        r = ff*s1[:,1] + ffn*s2[:,1]

        if self.validate(r.tolist()):
            print("valid", r.tolist())


    def solve_eq1(self, value):

        solver = pywrapcp.Solver("n-queens")


        #t1 = solver.IntVar(0, self.n-1, 't1')
        #t2 = solver.IntVar(0, self.n-1, 't2')

        #a*b*8 + c*4 + d = value

        maxv = int(value/5)
        a = solver.IntVar(2, maxv, 'a')
        b = solver.IntVar(2, maxv, 'b')
        c = solver.IntVar(2, maxv, 'c')
        d = solver.IntVar(0, maxv, 'd')

        q = [a, b, c, d]

        solver.Add(a>b)
        solver.Add(a>c)
        solver.Add(a>d)

        solver.Add(c>d)

        for i in range(2, int(math.sqrt(value))):
            if not is_prime(i):
                solver.Add(a != i)
                solver.Add(b != i)

        solver.Add(a*b*8 + c*4 + d == value)

        db = solver.Phase(q, solver.CHOOSE_MIN_SIZE_LOWEST_MAX,solver.ASSIGN_CENTER_VALUE)

        solver.Solve(db)

        solver.NewSearch(db)

        while solver.NextSolution():
            r = [v.Value() for v in q]
            print(r)


q = SuperQueens(7)
s = q.init_sols()
import nqueens

print(s)

qq = nqueens.n_queens(7).all_solutions
ts = [tuple(x) for x in s]
a = [l for l in qq if tuple(l) not in ts]

print(len(a), a)

