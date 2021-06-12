import nqueens
import math
import numpy as np
import time
import matplotlib.pyplot as plt

phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l
c1 = math.pow(phi, 2/(math.pi))


def get_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    return m, b

diff_mem = None
def sol_diff(src, dst):
    total = 0

    if diff_mem[src[0]][dst[0]] != 0:
        return diff_mem[src[0]][dst[0]]

    for s, d in zip(src[1], dst[1]):
        if d >= s:
            total += d - s
        else:
            total += len(src) - d + s

    diff_mem[src[0]][dst[0]] = total

    return total

all_sols = []
all_diffs = []


def found_best_path(n):
    global diff_mem

    sols = nqueens.n_queens(n).all_solutions
    diff_mem = np.zeros((len(sols), len(sols)))
    sols = [[i, sol] for i, sol in enumerate(sols)]
    work_sols = []
    #for i, sol in enumerate(sols):
        #if list(reversed(sol[1])) not in work_sols:

    sol = [7, [2, 5, 1, 4, 0, 3, 6]]
    c_sols = sols[:]
    c_sols.pop(7)
    found_all_paths(c_sols, [sol])
    work_sols.append(sol[1])

    r = []
    m = min(all_diffs)
    for s, d in zip(all_sols, all_diffs):
        if d == m:
            for ss in s:
                r.append(ss[1])

    print(len(sols), len(all_diffs), len(set(all_diffs)))

    return r


def found_all_paths(sols, final_sol, idiffs=[]):
    global num_sols

    if len(sols) == 0:
        all_sols.append(final_sol)
        all_diffs.append(sum(idiffs))
        return

    min_diff = math.inf
    diffs = []
    for i, s in enumerate(sols):
        if s not in final_sol:
            diff = sol_diff(final_sol[-1], s)
            diffs.append((i, diff))
            if min_diff > diff:
                min_diff = diff

    for i, d in diffs:
        if d == min_diff:
            c_sols = sols[:]
            sol = c_sols.pop(i)
            c_final_sol = final_sol[:]
            c_final_sol.append(sol)
            c_idiffs = idiffs[:]
            c_idiffs.append(d)
            found_all_paths(c_sols, c_final_sol, c_idiffs)

n=7
start_time = time.time()
path = np.array(found_best_path(n))
print("--- %s seconds ---" % (time.time() - start_time))


n = 7
a = [(math.pi*2*i)/n for i in range(n)]
r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]




def get_full_sol(values, radious, n):
    y = []
    x = []
    perimeter = radious*2*math.pi
    delta = perimeter/n
    points = []
    for i in range(n):
        points.append(-perimeter/2 + i*delta)

    print(radious, perimeter, points)

    xv = 0
    for i, value in enumerate(values[:-1]):
        x0 = xv
        y0 = points[value]
        x.append(x0)
        y.append(y0)

        xx = np.linspace(x0, x0+math.pi, 30)
        x1 = xx[-1]
        y1 = points[values[i+1]]
        m, b = get_equation((x0, y0), (x1, y1))
        yy = m*xx + b

        x.extend(xx[1:-1].tolist())
        y.extend(yy[1:-1].tolist())

        xv += math.pi

    return x, y


x, y = get_full_sol(path[:,0], r[0], n)

plt.plot(x, y)
plt.show()




