from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd


class SearchMonitor(pywrapcp.SearchMonitor):
    def __init__(self, solver, q,one_solution=False):
        pywrapcp.SearchMonitor.__init__(self, solver)
        self.q = q
        self.all_solutions = []
        self.n = len(self.q)
        self.one_solution = one_solution

    def AcceptSolution(self):
        qval = [self.q[i].Value() for i in range(self.n)]
        self.all_solutions.append(qval)

        return self.one_solution


def n_queens(n, one_solution=False):
    g_solver = pywrapcp.Solver("n-queens")
    q = [g_solver.IntVar(0, n - 1, "x%i" % i) for i in range(n)]

    g_solver.Add(g_solver.AllDifferent(q))
    g_solver.Add(g_solver.AllDifferent([q[i] + i for i in range(n)]))
    g_solver.Add(g_solver.AllDifferent([q[i] - i for i in range(n)]))

    db = g_solver.Phase(q, g_solver.CHOOSE_MIN_SIZE_LOWEST_MAX,g_solver.ASSIGN_CENTER_VALUE)

    monitor = SearchMonitor(g_solver, q, one_solution)
    g_solver.Solve(db, monitor)

    g_solver.NewSearch(db)

    while g_solver.NextSolution():
        pass

    g_solver.EndSearch()

    print("n: ", n)
    print("all_solutions:", len(monitor.all_solutions))
    print("WallTime:", g_solver.WallTime(), "ms")

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

'''
print(np.sum(monitor.unique_solutions, axis=0))


def comb(ll, stack=[], r=[]):
	if len(ll) == 0:
		r.append(stack)
		return stack

	for v in ll[0]:
		l = stack.copy()
		l.append(v)
		if len(ll) > 1:
			comb(ll[1:], l)
		else:
			comb([], l)

	return r


l = comb(g)
print(l)
print(np.shape(l))


o = [np.sum(x, axis=0) for x in l]
print(o)
'''

