from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd


class SearchMonitor(pywrapcp.SearchMonitor):
	def __init__(self, solver, q,one_solution=False):
		pywrapcp.SearchMonitor.__init__(self, solver)
		self.q = q
		self.all_solutions = []
		self.unique_solutions = []
		self.count_symmetries = [0]*7
		self.n = len(self.q)
		self.one_solution = one_solution

	def AcceptSolution(self):
		qval = [self.q[i].Value() for i in range(self.n)]
		self.all_solutions.append(qval)

		symmetries = [vv in self.unique_solutions for vv in gen_symmetries(self.n, qval)]
		self.count_symmetries = [i+v for i,v in zip(symmetries, self.count_symmetries)]

		if sum(symmetries) == 0:
			self.unique_solutions.append(qval)

		return self.one_solution


	def group_solutions(self):
		r = []

		for u in self.unique_solutions:
			l = gen_symmetries(self.n, u)
			l.append(u)
			r.append(list(map(list, list(set(map(tuple, l))))))
		return r

def gen_symmetries(n, solution):

	symmetries = []

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
	print("unique_solutions:" , len(monitor.unique_solutions))
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

#s = [count_sols(i) for i in range(7, 13)]


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

