from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd
import pickle
import os.path


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
		qval = [self.q[i].Value() for i in range(self.n)]
		self.all_solutions.append(qval)

		if self.unique_solution:
			symmetries = [vv in self.unique_solutions for vv in gen_symmetries(self.n, qval)]
			self.count_symmetries = [i+v for i,v in zip(symmetries, self.count_symmetries)]

			if sum(symmetries) == 0:
				self.unique_solutions.append(qval)

		return self.one_solution

	def RefuteDecision(self, d):
		qval = [self.q[i].Value() for i in range(self.n)]
		print('bad', qval)

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

	db = g_solver.Phase(q, g_solver.CHOOSE_MIN_SIZE_LOWEST_MAX,g_solver.ASSIGN_CENTER_VALUE)

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

		#for i, v in enumerate(sol[2]):
		#	if i not in sol[0]:
		#		g_solver.Add(q[i] != v)

	g_solver.Add(g_solver.AllDifferent(q))
	g_solver.Add(g_solver.AllDifferent([q[i] + i for i in range(n)]))
	g_solver.Add(g_solver.AllDifferent([q[i] - i for i in range(n)]))


	db = g_solver.Phase(q, g_solver.CHOOSE_MIN_SIZE_LOWEST_MAX,g_solver.ASSIGN_CENTER_VALUE)

	monitor = SearchMonitor(g_solver, q, one_solution, unique_solution)

	if os.path.exists('.' + str(n) + '_all') and sol is None:
		all_solutions = pickle.load(open('.' + str(n) + '_all', "rb"))
		monitor.all_solutions = all_solutions

		if os.path.exists('.' + str(n) + '_unique') and sol is None:
			unique_solutions = pickle.load(open('.' + str(n) + '_unique', "rb"))
			monitor.unique_solutions = unique_solutions
		return monitor

	g_solver.Solve(db, monitor)

	g_solver.NewSearch(db)

	while g_solver.NextSolution():
		pass

	g_solver.EndSearch()

	print("n: ", n)
	print("all_solutions:", len(monitor.all_solutions))
	print("unique_solutions:" , len(monitor.unique_solutions))
	print("WallTime:", g_solver.WallTime(), "ms")

	if sol is None and not one_solution and not os.path.exists('.' + str(n) + '_all'):
		pickle.dump(monitor.all_solutions, open('.' + str(n) + '_all', "wb"))
		if len(monitor.unique_solutions) > 0:
			pickle.dump(monitor.unique_solutions, open('.' + str(n) + '_unique', "wb"))

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




print(n_queens(7).all_solutions)