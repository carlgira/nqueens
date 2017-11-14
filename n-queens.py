from __future__ import print_function
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
		print(self._nexts)
		index, var = self.NextVar()

		if var:
			value = self.BounVar(var)

			decision = CustomDecision(var, index, value, self._nexts)
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

	def __init__(self, var, index, value, q):
		pywrapcp.PyDecision.__init__(self)
		self.__var = var
		self.__index = index
		self.__value = value
		self.__q = q

	def Apply(self, solver):
		self.__var.SetValue(self.__value)


		# x(r[i]=j) → r[n−i+1]=j
		#self.__q[n - 1 - self.__index].RemoveValue(self.__value)

		# r180(r[i]=j) → r[n−i+1]=n−j+1
		#self.__q[n - 1 - self.__index].RemoveValue((n - 1 -  self.__value))

		# r270(r[i]=j) → r[n−j+1]=i
		#solver.Add(self.__q[n - 1 - self.__value] != self.__index)


	def Refute(self, solver):
		self.__var.RemoveValue(self.__value)

	def DebugString(self):
		return('CustomDecision')


def main(n=8, num_sol=0, print_sol=1):
	# Create the solver.
	solver = pywrapcp.Solver("n-queens")

	# pywrapcp.PyDecisionBuilder()

	# monitor = pywrapcp.SearchMonitor(solver)
	# solver.DecisionBuilderFromAssignment()


	#
	# data
	#
	print("n:", n)

	# declare variables
	all_solutions = []
	q = [solver.IntVar(0, n - 1, "x%i" % i) for i in range(n)]

	#
	# constraints
	#


	solver.Add(solver.AllDifferent(q))
	solver.Add(solver.AllDifferent([q[i] + i for i in range(n)]))
	solver.Add(solver.AllDifferent([q[i] - i for i in range(n)]))

	for i in range( math.floor(n/2)):
		for j in range(n):
			q[n - 1 - i].RemoveValue(j)

	print(q)


	l1 = [n - 1] * n
	final_solutions = []

	class SearchMonitorTest(pywrapcp.SearchMonitor):
		def __init__(self, solver, nexts):
			print('Build')
			pywrapcp.SearchMonitor.__init__(self, solver)
			self._nexts = nexts

		def AtSolution(self):
			qval = [self._nexts[i].Value() for i in range(n)]

			symmetries = [
				[qval[n - i - 1] for i in range(n)],
				[n - (qval[i] + 1) for i in range(n)],
				# [qval.index(i) for i in range(n)],
				# [n - (qval.index(n-(i+1)) + 1) for i in range(n)],
				[n - i for i in range(n)],
				# [n - (qval[n-(i+1)] + 1) for i in range(n)],
				[qval.index(n - (i + 1)) for i in range(n)]
			]

			# x(r[i]=j) → r[n−i+1]=j
			# y(r[i]=j) → r[i]=n−j+1
			# d1(r[i]=j) → r[j]=i
			# d2(r[i]=j) → r[n−j+1]=n−i+i

			# r90(r[i]=j) → r[j] = n−i+1
			# r180(r[i]=j) → r[n−i+1]=n−j+1
			# r270(r[i]=j) → r[n−j+1]=i

			for symmetry in symmetries:
				if symmetry in final_solutions:
					return False
			final_solutions.append(qval)

			return True

	db = solver.Phase(q,
					  solver.CHOOSE_MIN_SIZE_LOWEST_MAX,
					  solver.ASSIGN_CENTER_VALUE)

	# monitor = SearchMonitorTest(solver, q)
	# solver.Solve(db, monitor)

	custom_db = CustomDecisionBuilder(q)

	composed_db = solver.Compose([db, custom_db])

	solver.Solve(custom_db)

	# l1 = list(range(n))
	# def check_symmetries(val):
	#	return np.subtract(l1, val).tolist() not in final_solutions

	solver.NewSearch(db)
	num_solutions = 0
	while solver.NextSolution():

		if print_sol:
			# print(q)
			qval = [q[i].Value() for i in range(n)]

		# if check_symmetries(qval):
		# final_solutions.append(qval)
		num_solutions += 1
		if num_sol > 0 and num_solutions >= num_sol:
			break

	solver.EndSearch()

	print("num_solutions:", num_solutions)
	print("unique_solutions:", len(final_solutions))
	print("failures:", solver.Failures())
	print("branches:", solver.Branches())
	print("WallTime:", solver.WallTime(), "ms")


n = 8
num_sol = 0
print_sol = 1
if __name__ == "__main__":
	if len(sys.argv) > 1:
		n = int(sys.argv[1])
	if len(sys.argv) > 2:
		num_sol = int(sys.argv[2])
	if len(sys.argv) > 3:
		print_sol = int(sys.argv[3])

	main(n, num_sol, print_sol)

# print_sol = False
# show_all = False
# for n in range(1000,1001):
#     print
#     main(n, num_sol, print_sol)
