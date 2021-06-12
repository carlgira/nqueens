from super_queens import SuperQueens
import nqueens
import numpy as np
import numpy.ma as ma
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model
import pickle

'''
    Solver to get solutions on combining a initial group.
    Not working as expected.
'''
def super_solver(n, li, lt, c=0):

    print(len(li), len(lt), c)

    if len(lt) == 0:
        print("Good")
        return 0

    if len(li) == 0 and len(lt) > 0:
        print("Bad")
        return 0

    nsols = {}
    for st in lt:
        cfilters = np.sum(li == st, axis=1)
        #print(cfilters)
        pos_f1 = np.where((cfilters > 1))
        for p1, cf1 in zip(pos_f1[0], cfilters[pos_f1]):
            pos_f2 = np.where((cfilters >= n-cf1))
            for p2, cf2 in zip(pos_f2[0], cfilters[pos_f2]):
                f1 = li[p1] == st
                f2 = li[p2] == st
                if p1 < c and p2 < c:
                    continue
                m1 = ma.masked_array(li[p1], mask=np.logical_not(f1), fill_value=0)
                m2 = ma.masked_array(li[p2], mask=np.logical_not(f2), fill_value=0)
                m3 = ma.masked_array(li[p1]*-1, mask=np.logical_not(f1 == f2), fill_value=0)
                sol = m1.filled() + m2.filled() + m3.filled()
                if sq.validate(sol):
                    if tuple(list(sol)) in nsols:
                        nsols[tuple(list(sol))].append([li[p1], li[p2], f1, f2, m3.mask])
                    else:
                        nsols[tuple(list(sol))] = [ [li[p1], li[p2], f1, f2, m3.mask]]

    ninit = []
    for a in nsols.keys():
        sym = nqueens.gen_symmetries(n, list(a))
        sym.append(list(a))
        for s in sym:
            if s not in ninit:
                ninit.extend(sym)

    r = []
    for sol in lt:
        sol = list(sol)
        if sol not in ninit:
            r.append(sol)

    other_sols = [np.array(o) for o in r]
    init_sols = [np.array(i) for i in ninit]
    init_sols.extend(li)
    c = len(li)

    super_solver(n, init_sols, other_sols, c)


    #print("sym")
    #print("all", ss.keys())
    #for d in ss.keys():
        #print(d, nqueens.gen_symmetries(n, d))

    return -1

'''
n = 11
sq = SuperQueens(n)

other_sols, init_sols = sq.other_sols(sq.init_sols(), nqueens.n_queens(n).all_solutions)
print("i", init_sols)
print("o",other_sols )
other_sols = [np.array(o) for o in other_sols]
init_sols = [np.array(i) for i in init_sols]

print(len(init_sols), len(other_sols))

super_solver(n, init_sols, other_sols)
'''




'''
    Function to compare to arrays. 
    arr1 can be bigger that arr2 and compare it in a circular manner.
'''
def array_in(arr1, arr2):
    n = len(arr1)
    m = len(arr2)
    if m == n:
        return np.equal(arr1, arr2)
    arr1o = np.concatenate([arr1, arr1[:-1]])
    for i in range(len(arr1o)-m):
        if np.array_equal(arr2, arr1o[i:i+m]):
            return True
    return False


'''
    Compare if there solution of one n are related with previous n. 
    The results is that is ONLY related exactly with previous one (n-1)
    Meaning, that there are some solutions on n=8 in n=7, and that there are some solutions of n=9 on n=8
'''
def compare_sols(n):
    osols = [np.array(sol) for sol in nqueens.n_queens(n).all_solutions]

    print(n, len(osols))
    for ne in range(5, n-1):

        sols = [np.array(sol) for sol in nqueens.n_queens(ne).all_solutions]
        print(ne, len(sols))
        for sol in sols:
            for i in range(len(sols)):
                if array_in(osols[i], sol):
                    print("in", sol, osols[i])

#compare_sols(13)

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, isols, masks,  osols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__isols = isols
        self.__masks = masks
        self.__osols = osols
        self.__solution_count = 0
        self.all_solutions = []

    def OnSolutionCallback(self):
        self.__solution_count += 1
        solution = []
        for i, m, o in zip(self.__isols, self.__masks, self.__osols):
            solution.append([self.Value(i), self.Value(m), self.Value(o)])
        self.all_solutions.append(solution)

    def SolutionCount(self):
        return self.__solution_count


def search_posible_solutions(gdata, numof_inputs, numof_igroups , numof_ogroups, numof_mask, numof_solutions):
    model = cp_model.CpModel()

    inputs = [model.NewIntVar(0, numof_inputs-1, 'in_%i' % i) for i in range(numof_ogroups)]
    #igroups = [model.NewIntVar(0, numof_igroups-1, 'ig_%i' % i) for i in range(numof_ogroups)]
    solutions = [model.NewIntVar(0, numof_solutions-1, 'sn_%i' % i) for i in range(numof_ogroups)]
    masks = [model.NewIntVar(0, numof_mask-1, 'ma_%i' % i) for i in range(numof_ogroups)]

    #model.AddAllDifferent(inputs)
    model.AddAllDifferent(solutions)

    for i, sol in enumerate(gdata):
        model.AddAllowedAssignments([inputs[i], masks[i], solutions[i]], sol)

    solver = cp_model.CpSolver()

    solution_printer = SolutionPrinter(inputs, masks, solutions)
    solver.SearchForAllSolutions(model, solution_printer)

    #solver.Solve(model)
    #solution_printer.all_solutions = []
    #for i, m, o in zip(inputs, masks, solutions):
    #    solution_printer.all_solutions.append([solver.Value(i), solver.Value(m), solver.Value(o)])
    #solution_printer.__solution_count = len(solution_printer.all_solutions)

    print('Solutions found : %i' % solution_printer.SolutionCount())
    #for sol in solution_printer.all_solutions:
    #    print(sol)

    return solution_printer.all_solutions


def preprocess_similar(n, sim_data):
    isols = sim_data[:, 0]
    masks = sim_data[:, 1]
    osols = sim_data[:, 2]

    u_isols = np.unique(isols, axis=0)
    u_osols = np.unique(osols, axis=0)
    u_masks = np.unique(masks, axis=0)

    igroups = []
    for e in range(len(isols)):
        isol = isols[e].tolist()
        added = False
        syms = nqueens.gen_symmetries(n, isol)
        for i in range(len(igroups)):
            if isol in igroups[i]:
                added = True
                break
            elif igroups[i][0] in syms:
                igroups[i].append(isol)
                added = True
                break
        if not added:
            igroups.append([isol])

    igroups = np.array(igroups)

    ogroups = []
    result = {}
    for e in range(len(osols)):
        sol = osols[e].tolist()
        added = False
        syms = nqueens.gen_symmetries(n, sol)
        for i in range(len(ogroups)):
            if sol in ogroups[i]:
                added = True
                result[i].append([np.where(np.sum(u_isols == isols[e], axis=1) == n)[0][0],
                                 # np.where([len(np.where(np.sum(igroups[i] == isols[e], axis=1) == n)[0]) for i in range(len(igroups))])[0][0],
                                  np.where(np.sum(u_masks == masks[e], axis=1) == n)[0][0],
                                  np.where(np.sum(u_osols == osols[e], axis=1) == n)[0][0]])
                break
            elif ogroups[i][0] in syms:
                ogroups[i].append(sol)
                added = True
                result[i].append([np.where(np.sum(u_isols == isols[e], axis=1) == n)[0][0],
                                #  np.where([len(np.where(np.sum(igroups[i] == isols[e], axis=1) == n)[0]) for i in range(len(igroups))])[0][0],
                                  np.where(np.sum(u_masks == masks[e], axis=1) == n)[0][0],
                                  np.where(np.sum(u_osols == osols[e], axis=1) == n)[0][0]])
                break
        if not added:
            ogroups.append([sol])
            result[len(ogroups) - 1] = [[np.where(np.sum(u_isols == isols[e], axis=1) == n)[0][0],
                                 #        np.where([len(np.where(np.sum(igroups[i] == isols[e], axis=1) == n)[0]) for i in range(len(igroups))])[0][0],
                                    np.where(np.sum(u_masks == masks[e], axis=1) == n)[0][0],
                                    np.where(np.sum(u_osols == osols[e], axis=1) == n)[0][0]]]

    return u_isols, u_osols, u_masks , len(u_isols), len(u_osols), len(u_masks), len(igroups), len(ogroups), list(result.values())

'''
    Get the transformations of trivial solutions and tried to find a possible path using CP solver to get a strategy
'''
def find_solver(n):
    sq = SuperQueens(n)
    gg = sq.check_all_similar_init()
    sq.check_most_similar_init() # 88 2592 {4: 996, 5: 892, 6: 520, 7: 140, 8: 24, 9: 20}

    u_isols, u_osols, u_masks, u_isols_count, u_osols_count, u_masks_count, igroups_count, ogroups_count, result = preprocess_similar(n, np.array(gg))

    sols = search_posible_solutions(result, u_isols_count, igroups_count , ogroups_count, u_masks_count, u_osols_count)

    data = {}
    data['gg'] = gg
    data['u_isols'] = u_isols
    data['u_masks'] = u_masks
    data['u_osols'] = u_osols
    data['igroups_count'] = igroups_count
    data['ogroups_count'] = ogroups_count
    data['u_masks_count'] = u_masks_count
    data['result'] = result
    data['sols'] = sols

    pickle.dump(data, open('solver/' + str(n) + '.pck', "wb"))


data = pickle.load(open('solver/' + str(8) + '.pck', "rb" ))


print(data['igroups_count'], data['ogroups_count'], data['u_masks_count'])

print(data['sols'][0:10])




#search_posible_solutions( [  [[0, 1, 1],[1, 2, 2],[2, 1, 0]], [[0, 1, 2],[1, 3, 1],[2, 1, 0], [3, 2, 1]], [[0, 0, 0]]  ], 4, 3, 3)
