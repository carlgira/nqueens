from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd
import pickle
import os.path
import math
import nqueens


class SearchMonitor(pywrapcp.SearchMonitor):
    def __init__(self, solver, q,one_solution=False, unique_solution=True):
        pywrapcp.SearchMonitor.__init__(self, solver)
        self.q = q
        self.all_solutions = []
        self.unique_solutions = []
        self.count_symmetries = [0]*7
        self.n = len(self.q)
        self.one_solution = one_solution
        self.unique_solution = unique_solution

    def AcceptSolution(self):
        print('accept')
        qval = [self.q[i].Value() for i in range(self.n)]
        self.all_solutions.append(qval)

        if self.unique_solution:
            symmetries = [vv in self.unique_solutions for vv in gen_symmetries(self.n, qval)]
            self.count_symmetries = [i+v for i,v in zip(symmetries, self.count_symmetries)]

            if sum(symmetries) == 0:
                self.unique_solutions.append(qval)

        return self.one_solution

    def EndNextDecision(self, d, b):
        print('r', self.q, b)


    def BeginFail(self):
        print('back', self.q)

    def group_solutions(self):
        r = []

        for u in self.unique_solutions:
            l = gen_symmetries(self.n, u)
            l.append(u)
            r.append(list(map(list, list(set(map(tuple, l))))))
        return r

def gen_symmetries(n, solution):

    symmetries = []

    #x(r[i]=j) → r[n−i+1]=j
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

def get_star_sols(n):
    center_value = n//2
    g_solver = pywrapcp.Solver("n-queens")

    q = [g_solver.IntVar(0, n - 1, "x%i" % i) for i in range(n)]
    q[center_value] = g_solver.IntVar(center_value, center_value, "x%i" % center_value)

    g_solver.Add(g_solver.AllDifferent(q))
    g_solver.Add(g_solver.AllDifferent([q[i] + i for i in range(n)]))
    g_solver.Add(g_solver.AllDifferent([q[i] - i for i in range(n)]))
    for i in range(center_value):
        g_solver.Add(q[i] + q[-i-1] == n-1)

    db = g_solver.Phase(q, g_solver.CHOOSE_MIN_SIZE,g_solver.ASSIGN_MIN_VALUE)

    g_solver.Solve(db)

    g_solver.NewSearch(db)

    sols = []
    nsols = 0
    while g_solver.NextSolution():
        nsols +=1
        pass

    print(n, nsols)
    print("WallTime:", g_solver.WallTime(), "ms")

    g_solver.EndSearch()

def n_queens(n, sol=None ,one_solution=False, unique_solution=True):
    g_solver = pywrapcp.Solver("n-queens")

    q = [g_solver.IntVar(0, n - 1, "x%i" % i) for i in range(n)]

    if sol is not None and len(sol) > 0:
        for l, m, i in zip(sol[0], sol[1], range(n)):
            q[l] = g_solver.IntVar(m, m, "x%i" % i)

    g_solver.Add(g_solver.AllDifferent(q))
    g_solver.Add(g_solver.AllDifferent([q[i] + i for i in range(n)]))
    g_solver.Add(g_solver.AllDifferent([q[i] - i for i in range(n)]))


    db = g_solver.Phase(q, g_solver.CHOOSE_MIN_SIZE,g_solver.ASSIGN_MIN_VALUE)

    monitor = SearchMonitor(g_solver, q, one_solution, unique_solution)

    g_solver.Solve(db, monitor)

    g_solver.NewSearch(db)

    while g_solver.NextSolution():
        pass

    g_solver.EndSearch()

    return monitor


def print_board(b):
    r = []
    for i in range(len(b)):
        l = []
        for j in range(len(b)):
            if b[j] == i:
                l.append('X')
            else:
                l.append('_')
        r.append(l)
    print(np.array(r))


def count_sols(n):
    mon = n_queens(n)
    psols = pd.DataFrame(mon.all_solutions)
    r = []
    for i in range(n):
        v = np.array(psols.groupby(i).count().values)
        r.append(v[:,0].tolist())
    return np.array(r)

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




n = 7
phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l


c1 = math.pow(phi, 2/(math.pi))
f = [(2*i)/n for i in range(n)]

a = [(math.pi*2*i)/n for i in range(n)]
r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

board = []

# Complex Circular board
m = (np.outer(r, np.cos(a)) + 1j*np.outer(r, np.sin(a))).T

sols = nqueens.n_queens(n).all_solutions

def calc_area(q):
    qcount = np.sum([1 for v in q if q.Size() == 1])


