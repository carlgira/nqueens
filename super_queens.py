import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fractions import Fraction
import math
import itertools
import nqueens
from collections import Counter
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

def polar_2_cartesian(theta, r, x0=0.0, y0=0.0):
    return r*np.cos(theta) + x0 , r*np.sin(theta) + y0


class Spiral2D:
    def __init__(self):
        self.x0 = 0.0
        self.y0 = 0.0


    def draw(self, c1, c2, sol):
        n = len(sol)
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
        ax = plt.subplot2grid((1, 1), (0, 0))

        # Circles of principal
        for rr in r:
            theta = np.pi * np.arange(0, 2, 0.01)
            X, Y = polar_2_cartesian(theta, [rr]*len(theta), self.x0, self.y0)
            #ax.plot(X, Y)

        # Circles of sencondary
        for rr in rn:
            theta = np.pi * np.arange(0, 2, 0.01)
            X, Y = polar_2_cartesian(theta, [rr]*len(theta), self.x0, self.y0)
            #ax.plot(X, Y)

        for aa, ff in zip(a, f):
            theta1 = np.pi * np.arange(0, 12 - ff, 0.01)
            radious1 = [spiral1(c1, t, aa) for t in theta1]
            radious1, theta1 = clean_data(radious1, theta1, r[0])

            theta2 = np.pi * np.arange(2 - ff, 12, 0.01)
            radious2 = [spiral2(c1, t, aa) for t in theta2]
            radious2, theta2 = clean_data(radious2, theta2, r[0], False)

            X, Y = polar_2_cartesian(theta1, radious1, self.x0, self.y0)
            #ax.plot(X, Y, c='b')

            X, Y = polar_2_cartesian(theta2, radious2, self.x0, self.y0)
            #ax.plot(X, Y, c='r')

            ts1 = [spiral1_inv(c1, t, aa) for t in r]
            X, Y = polar_2_cartesian(ts1, r, self.x0, self.y0)
            #ax.scatter(X, Y, s=20, c='y')

            ts2 = [spiral1_inv(c1, t, aa) for t in rn]
            X, Y = polar_2_cartesian(ts2, rn, self.x0, self.y0)
            #ax.scatter(X, Y, s=20, c='g')

        tt = [a[v] for v in sol]
        X0, Y0 = polar_2_cartesian(tt, r, self.x0, self.y0)
        ax.scatter(X0, Y0, s=30, c='black')

        n = n//2 -1
        f = [(2*i)/n for i in range(n)]
        a = [(math.pi*2*i)/n for i in range(n)]
        r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

        an = [(math.pi*2*i)/n for i in range(n)]
        rn = [spiral1(c1, 10*math.pi + math.pi/n, aa) for aa in an]

        a = np.array(a)
        r = np.array(r)

        an = np.array(an)
        rn = np.array(rn)
        for i in range(len(a)):
            theta_min = a[i] - 12*math.pi - math.log(r[0], c2)
            theta3 = np.pi * np.arange(theta_min/math.pi, 12 - f[i], 0.01)
            radious3 = [spiral1(c2, t, a[i]) for t in theta3]
            radious3, theta3 = clean_data(radious3, theta3, r[0])

            X, Y = polar_2_cartesian(theta3, radious3, X0[len(X0)//2], Y0[len(Y0)//2])
            ax.plot(X, Y, c='blue')

            theta_max = 12*math.pi - a[i] - math.log(r[-1], c2)
            theta4 = np.arange(2*math.pi - a[i], theta_max, 0.01)
            radious4 = [spiral2(c2, t, an[i]) for t in theta4]
            radious4, theta4 = clean_data(radious4, theta4, r[0], False)

            X, Y = polar_2_cartesian(theta4, radious4, X0[len(X0)//2], Y0[len(Y0)//2])
            ax.plot(X, Y, c='red')

        plt.show()


'''
        theta_min = a[0] - 12*math.pi - math.log(r[0], c2)
        theta3 = np.pi * np.arange(theta_min/math.pi, 12 - f[0], 0.01)
        radious3 = [spiral1(c2, t, a[0]) for t in theta3]
        radious3, theta3 = clean_data(radious3, theta3, r[0])

        X, Y = polar_2_cartesian(theta3, radious3, X0[n//2], Y0[n//2])
        ax.plot(X, Y, c='blue')

        theta_max = 12*math.pi - a[0] - math.log(r[-1], c2)
        theta4 = np.arange(2*math.pi - a[5], theta_max, 0.01)
        radious4 = [spiral2(c2, t, an[5]) for t in theta4]
        radious4, theta4 = clean_data(radious4, theta4, r[0], False)

        X, Y = polar_2_cartesian(theta4, radious4, X0[n//2], Y0[n//2])
        ax.plot(X, Y, c='red')
'''




#spiral = Spiral2D()
#spiral.draw(math.pow(phi, 2/(math.pi)), math.pow(phi, 1/(4*math.pi)), [4,8,0,9,3,5,0,0,0,0,0])





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

        n = len(sol)-1
        if len(sol) != self.n:
            return False

        if np.sum(sol) != n*(n+1)/2:
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


    def init_sols(self, l=None):
        max_c = int(self.n/2)
        if l is not None:
            max_c = l
        sols = []

        for d in range(1, max_c):
            c = math.pow(phi, 1/(d*math.pi))
            for i, a in enumerate(self.a):
                s1, s2 = self.get_sol(c, a)
                sols.append(s1[:,1])
                sols.append(s2[:,1])

        return sols

    def init_sols_label(self):
        max_c = int(self.n/2)
        sols = []

        for d in range(1, max_c):
            c = math.pow(phi, 1/(d*math.pi))
            for i, a in enumerate(self.a):
                s1, s2 = self.get_sol(c, a)
                sols.append([s1[:,1], d, i])
                sols.append([s2[:,1], -d, i])

        return sols

    def groups_init_sols(self):
        max_c = int(self.n/2)
        sols1 = []
        sols2 = []

        for d in range(1, max_c):
            c = math.pow(phi, 1/(d*math.pi))
            sol1 = []
            sol2 = []
            for i, a in enumerate(self.a):
                s1, s2 = self.get_sol(c, a)
                sol1.append(list(s1[:,1]))
                sol2.append(list(s2[:,1]))
            sols1.append(sol1)
            sols2.append(sol2)

        return sols1, sols2

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


    def get_sols(self, c1):
        sols = []
        for i, a in enumerate(self.a):
            s1, s2 = self.get_sol(c1, a)
            sols.append(s1[:,1].tolist())
            sols.append(s2[:,1].tolist())
        return sols

    def get_all_sols(self):
        result = []
        for v in range(1, math.ceil(self.n/2)):
            c1 = math.pow(phi, v/(math.pi))
            result.extend(self.get_sols(c1))
        return result

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


                all_filters = self.all_filters()
                for f in all_filters:
                    for ff in f:

                        ffn = np.logical_not(ff)
                        rr = ff*s1[:,1] + ffn*s2[:,1 ]

                        if self.validate(rr.tolist()) and not np.logical_and(filter, ff).any():
                            sss.append(r)
                            sols.append(tuple(rr.tolist()))

        sols = [list(s) for s in set(sols)]

        sss.sort()

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

    '''
        Other solutions out of the trivial ones of the spirals
    '''
    def other_sols(self, init_sols, all_sols):
        init_sols = [list(sol) for sol in init_sols]
        r = []
        for sol in all_sols:
            if sol not in init_sols:
                r.append(sol)
        return r, init_sols

    '''
        Stats of how similar is a regular solution against the trivial spiral ones
    '''
    def check_most_similar_init(self):
        other_sols, init_sols = self.other_sols(self.init_sols(), nqueens.n_queens(self.n).all_solutions)
        other_sols = [np.array(o) for o in other_sols]
        init_sols = [np.array(i) for i in init_sols]
        return self.check_most_similar(other_sols, init_sols)

    '''
        Stats of how similar is a regular solution against the spiral ones
    '''
    def check_most_similar(self, other_sols, init_sols):

        if len(other_sols) == 0:
            return {}, {}
        d = {}
        g = {}
        for o in range(self.n):
            g[o] = []

        for osol in other_sols:
            m = -1
            f = []
            s = None
            for isol in init_sols:
                v = np.sum(osol == isol)
                if m < v:
                    m = v
                    f = osol == isol
                    s = isol
            d[tuple(osol)] = m
            g[m].append([s, f])

        values = list(d.values())
        unique, counts = np.unique(values, return_counts=True)
        dd = dict(zip(unique, counts))
        ddd = {}
        for k in dd.keys():
            ddd[k] = dd[k]/len(other_sols)


        gg = {}
        for k in g.keys():
            if len(g[k]) > 0:
              gg[k] = g[k]

        '''
        if len(list(gg.values())[-1]) > 0:
            #init_sols.extend(list(gg.values())[-1])
            #init_sols = list(gg.values())[-1]
            #init_sols = [np.array(o) for o in init_sols]
            #other_sols = [list(o) for o in other_sols]
            #oo, _ = self.other_sols(init_sols, other_sols)
            #oo = [np.array(o) for o in oo]
            ma = np.argmax(list(dd.values()))
            init_sols.extend(list(gg.values())[ma])
            oo, _ = self.other_sols(init_sols, nqueens.n_queens(self.n).all_solutions)
            oo = [np.array(o) for o in oo]
            self.check_most_similar(oo, init_sols)
        '''

        return dd, ddd

    '''
        Stats of how similar is a regular solution against the trivial spiral ones
    '''
    def check_all_similar_init(self):
        all_solutions = nqueens.n_queens(self.n).all_solutions
        other_sols, init_sols = self.other_sols(self.init_sols(), all_solutions)
        other_sols = [np.array(o) for o in other_sols]
        init_sols = [np.array(i) for i in init_sols]
        return self.check_all_similar(other_sols, init_sols)

    '''
            Stats of all possible transformations of the init to the other solutions
    '''
    def check_all_similar(self, other_sols, init_sols, i=0):

        val = 4
        if len(other_sols) == 0:
            return {}, {}
        d = {}
        g = []
        m = []

        for isol in init_sols[i:]:
            d[tuple(isol)] = []
            for osol in other_sols:
                v = np.sum(osol == isol)
                if v == self.n-val:
                    m.append([isol, isol == osol, osol])

                    d[tuple(isol)].append([osol, isol, osol == isol])
                    if len(g) == 0 or len(np.where(np.sum(g == osol, axis=1) == self.n)[0]) == 0:
                        g.append(osol)
                        syms = np.array(nqueens.gen_symmetries(self.n, osol))
                        for sym in syms:
                            if len(np.where(np.sum(other_sols == sym, axis=1) == self.n)[0]) > 0 and \
                                    len(np.where(np.sum(g == sym, axis=1) == self.n)[0]) == 0:
                                g.append(sym)


        #values = d.values()
        #unique, counts = np.unique(values, return_counts=True)
        #dd = dict(zip(unique, counts))

        if len(g) == 0:
            return

        if len(g) > 0:
            isize = len(init_sols)
            init_sols.extend(g)
            oo, _ = self.other_sols(init_sols, nqueens.n_queens(self.n).all_solutions)
            oo = [np.array(o) for o in oo]
            #self.check_all_similar(oo, init_sols, isize)

        return m

    def check_star_sols(self):
        q = nqueens.n_queens(self.n)
        init_sols = self.init_sols()
        all_sols = q.all_solutions
        other_sols, _ = self.other_sols(init_sols, all_sols)
        other_sols = [np.array(sol) for sol in other_sols]
        init_sols_labels = self.init_sols_label()

        def check_sol(sol):
            n = len(sol)
            val = n//2
            if sol[val] != val:
                return False
            return ((np.flip(sol[:val]) + sol[val+1:]) == n-1).all()

        center_sol_i = [sol for sol in init_sols_labels if check_sol(sol[0])]
        center_sol_o = [sol for sol in other_sols if check_sol(sol)]

        return center_sol_i, center_sol_o

    def maks_of_others(self):
        q = nqueens.n_queens(self.n)
        s = SuperQueens(self.n)
        init_sols = s.init_sols()
        all_sols = q.all_solutions
        other_sols, _ = self.other_sols(init_sols, all_sols)
        other_sols = [np.array(sol) for sol in other_sols]
        init_sols_labels = s.init_sols_label()

        center_sol_i, center_sol_o = self.check_star_sols()

        masks = []
        groups = []

        for osol in other_sols:
            for isol in init_sols_labels:
                if np.sum(osol == isol[0]) >= self.n - 2:
                    masks.append([ (osol == isol[0]).astype(int).tolist(), isol[0].tolist(), osol.tolist(), isol[1], isol[2]])

        for e, m in enumerate(masks):
            sol = m[2]
            syms = nqueens.gen_symmetries(self.n, sol)
            syms.append(sol)
            flag = False
            for i, group in enumerate(groups):
                check = [sym for sym in syms if sym in group]
                if len(check) > 0:
                    flag = True
                    masks[e].append(i)
                    groups[i].append(sol)

            if not flag:
                masks[e].append(len(groups))
                groups.append([sol])

        masks.sort(key=lambda e: e[3])

        groups = []
        gmasks = []

        for m in masks:
            sol = m[2]
            syms = nqueens.gen_symmetries(self.n, sol)
            syms.append(sol)
            flag = False
            for i, group in enumerate(groups):
                check = [sym for sym in syms if sym in group]
                if len(check) > 0:
                    flag = True
                    groups[i].append(sol)
                    gmasks[i].append(m[0])
                    break
            if not flag:
                groups.append([sol])
                gmasks.append([m[0]])

        '''
        
        flag = False
        r = []
        for i in range(len(masks)):
            for e in range(i+1, len(masks)):
                if np.all(masks[i][0] | masks[e][0]):
                    m1 = np.copy(masks[i][2])
                    m1[masks[e][0]] = masks[e][1][[masks[e][0]]]
                    if self.validate(m1.tolist()):
                        r.append(m1)

        o = []
        for other in other_sols:
            if not (np.sum(np.equal(r, other), axis=1) == self.n).any():
                o.append(other)
        '''

        #return r #, o

    def get_masks(self, sol):
        s = SuperQueens(self.n)
        init_sols = s.init_sols()

        masks = []
        for isol in init_sols:
                masks.append([np.sum(sol == isol), sol == isol, isol, sol])

        return masks

    def poly_info(self):
        sols = nqueens.n_queens(self.n).all_solutions

        c = math.pow(phi, 2/(math.pi))
        a = [(math.pi*2*i)/n for i in range(n)]
        r = [spiral1(c, 10*math.pi + math.pi*2/n, aa) for aa in a]

        a = np.array(a)
        r = np.array(r)

        init_sols = self.init_sols()
        other_sols, _ = self.other_sols(init_sols, sols)
        other_sols = [np.array(sol) for sol in other_sols]


        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x,y
            else:
                return False

        def distance(p1,p2):
            x1, y1 = p1
            x2, y2 = p2
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return dist

        def perimeter(x, y):
            r = distance([x[0], y[0]], [x[-1], y[-1]])
            for i in range(len(x)-1):
                r += distance([x[i], y[i]], [x[i+1], y[i+1]])
            return r

        def PolyArea(x, y):
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


        def calc(sols):
            sums_angles = []
            min_angles = []
            max_angles = []
            permimeters = []
            for index, sol in enumerate(sols):
                tt = [a[v] for v in sol]
                X, Y = polar_2_cartesian(tt, r)
                angles = []

                for e in range(0, self.n):
                    ia = -1
                    ic = -1
                    if e == 0:
                        ia = np.where(sol == self.n-1)[0][0]
                    else:
                        ia = np.where(sol == e-1)[0][0]
                    ib = np.where(sol == e)[0][0]

                    if e == self.n-1:
                        ic = np.where(sol == 0)[0][0]
                    else:
                        ic = np.where(sol == e+1)[0][0]

                    A = np.array([X[ia], Y[ia]])
                    B = np.array([X[ib], Y[ib]])
                    C = np.array([X[ic], Y[ic]])

                    ba = A - B
                    bc = C - B
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(cosine_angle))

                    L1 = line(A, C)
                    L2 = line([0,0], B)

                    R = intersection(L1, L2)
                    if  distance([0,0], B) < distance([0,0], R):
                        angle = 360 - angle
                    angles.append(angle)

                permimeters.append(round(PolyArea(X, Y), 5))
                sums_angles.append(round(sum(angles)))
                min_angles.append(round(min(angles), 5))
                max_angles.append(round(max(angles), 5))

            sorted(sums_angles)
            sorted(min_angles)
            sorted(max_angles)

            return sums_angles, min_angles, max_angles, permimeters

        all_sols = [np.array(sol) for sol in sols]
        sums_angles, min_angles, max_angles, permimeters = calc(all_sols)

        all_permutations = [np.array(list(sol)) for sol in list(itertools.permutations(list(range(self.n))))]
        sums_angles, min_angles, max_angles, permimeters = calc(all_permutations)
