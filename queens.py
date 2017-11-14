import sys
from ortools.constraint_solver import pywrapcp
import numpy as np
import math



class CustomDecisionBuilder(pywrapcp.PyDecisionBuilder):
	def __init__(self, nexts):
		pywrapcp.PyDecisionBuilder.__init__(self)
		self.__done = False
		self._nexts = nexts
		self.__decisions = []

	def Next(self, solver):
		index, var = self.NextVar()

		if var:
			value = self.BounVar(var)

			decision = CustomDecision(var, index, value)
			self.__decisions.append(decision)
			return decision

		return None

	def BounVar(self, var):
		return var.Min()

	def NextVar(self):
		for i in range(n):
			if not self._nexts[i].Bound():
				return i, self._nexts[i]
		return None, None


class CustomDecision(pywrapcp.PyDecision):

	def __init__(self, var, index, value):
		pywrapcp.PyDecision.__init__(self)
		self.__var = var
		self.__index = index
		self.__value = value


	def Apply(self, solver):
		print("Apply", q, self.__index, self.__value)
		self.__var.SetValue(self.__value)


		#RemoveSymmetries(solver, self.__index, self.__value)




	def Refute(self, solver):
		print("Refute" , q, self.__index, self.__value)
		self.__var.RemoveValue(self.__value)


class SearchMonitorTest(pywrapcp.SearchMonitor):
	def __init__(self, solver, nexts):
		print('Build')
		pywrapcp.SearchMonitor.__init__(self, solver)
		self._nexts = nexts
		self.num_solutions = 0
		self.i = -1

	def AtSolution(self):
		qval = [self._nexts[i].Value() for i in range(n)]
		self.num_solutions += 1
		print(qval)

		return True


tt = []
def RemoveSymmetries(solver, index, value):

	if (index, value) in tt:
		return

	tt.append((index,value))

	#x(r[i]=j) → r[n−i+1]=j
	solver.Add(q[n - 1 - index] != value)

	#y(r[i]=j) → r[i]=n−j+1
	solver.Add(q[index] != (n - 1 - value))

	#d1(r[i]=j) → r[j]=i
	solver.Add(q[value] != index)


	# d2(r[i]=j) → r[n−j+1]=n−i+1
	solver.Add(q[n - 1 - value] != (n - 1 - index))


	# r90(r[i]=j) → r[j] = n−i+1
	solver.Add(q[value] != (n - 1 - index))

	# r180(r[i]=j) → r[n−i+1]=n−j+1
	solver.Add(q[n - 1 - index] != (n - 1 - value))

	# r270(r[i]=j) → r[n−j+1]=i
	solver.Add(q[n - 1 - value] != index)





n = 5

g_solver = pywrapcp.Solver("n-queens")
q = [g_solver.IntVar(0, n - 1, "x%i" % i) for i in range(n)]

g_solver.Add(g_solver.AllDifferent(q))
g_solver.Add(g_solver.AllDifferent([q[i] + i for i in range(n)]))
g_solver.Add(g_solver.AllDifferent([q[i] - i for i in range(n)]))

custom_db = CustomDecisionBuilder(q)

monitor = SearchMonitorTest(g_solver, q)
g_solver.Solve(custom_db, monitor)

g_solver.NewSearch(custom_db)

while g_solver.NextSolution():
	pass

g_solver.EndSearch()

print("n: ", n)
print("num_solutions:", monitor.num_solutions)

print("failures:", g_solver.Failures())
print("branches:", g_solver.Branches())
print("WallTime:", g_solver.WallTime(), "ms")





