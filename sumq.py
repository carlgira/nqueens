from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd

class SearchMonitor(pywrapcp.SearchMonitor):
	def __init__(self, solver,n, all_vars):
		pywrapcp.SearchMonitor.__init__(self, solver)
		self.all_vars = all_vars
		self.n = n
		self.count = 0

	def AcceptSolution(self):

		r = []
		for i in range(self.n):
			l = []
			for j in range(self.n):
				l.append(self.all_vars[i*self.n + j].Value())
			r.append(l)

		#print(np.array(r))
		self.count += 1

		return False


def n_queens_sum(n, value):
	g_solver = pywrapcp.Solver("n-queens")

	grid = {}
	for i in range(n):
		for j in range(n):
			grid[(i, j)] = g_solver.IntVar(1, 8, 'x%i%i' % (i, j))

	for i in range(n):
		g_solver.Add(g_solver.SumEquality([grid[(i, j)] for j in range(n)], value))
		g_solver.Add(g_solver.SumEquality([grid[(j, i)] for j in range(n)], value))

	for i in range(n):
		for j in range(n):
			g_solver.Add(grid[(i,j)] == grid[(j,i)]) # Columnas y filas iguales
			g_solver.Add(grid[(i,j)] == grid[(i,n-1-j)]) #
			g_solver.Add(grid[(j,i)] == grid[(n-1-j, i)])

			if n % 2 == 1 and i != int(n/2) and j != int(n/2):
				g_solver.Add(grid[(int(n/2),int(n/2))] != grid[(i,j)]) #


	all_vars = [grid[(i, j)] for i in range(n) for j in range(n)]

	db = g_solver.Phase(all_vars, g_solver.INT_VAR_SIMPLE, g_solver.INT_VALUE_SIMPLE)

	monitor = SearchMonitor(g_solver, n, all_vars)
	g_solver.Solve(db, monitor)

	g_solver.NewSearch(db)

	while g_solver.NextSolution():
		pass

	print("Sols", monitor.count)
	g_solver.EndSearch()

	return monitor


n_queens_sum(7, 40)