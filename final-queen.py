import nqueens
import math
import numpy as np

def gen_symmetries(solution, filter=None):

    symmetries = [solution]
    n = len(solution)

    x = list(range(n))
    y = list(range(n))
    d1 = list(range(n))
    d2 = list(range(n))
    r90 = list(range(n))
    r180 = list(range(n))
    r270 = list(range(n))

    for index in range(n):
        x[n - 1 - index] = solution[index]
        y[index] = (n - 1 - solution[index])
        d1[solution[index]] = index
        d2[n - 1 - solution[index]] = (n - 1 - index)
        r90[solution[index]] = (n - 1 - index)
        r180[n - 1 - index] = (n - 1 - solution[index])
        r270[n - 1 - solution[index]] = index

    symmetries.append(x)
    symmetries.append(y)
    symmetries.append(d1)
    symmetries.append(d2)
    symmetries.append(r90)
    symmetries.append(r180)
    symmetries.append(r270)

    if filter is not None:
        symmetries = np.array(symmetries)
        symmetries = symmetries[np.where(filter)].tolist()

    return symmetries


s = gen_symmetries([0, 2, 4, 6, 1, 3, 5], [True, False, True, False, True, True, False])


def queen_solver(n):

    sols = []

    sol0 = [2*i if i <= n/2 else int(2*(i-n/2))  for i in range(1, n)]
    sol1 = [2*i-1 if i <= n/2 else (2*(i-int(n/2)))  for i in range(1, n)]
    sol1.insert(int(n/2), 0)

    s1 = sol0.copy()
    s2 = sol0.copy()
    s1.append(0)
    s2.insert(0,0)

    sols.extend(gen_symmetries(s1))
    sols.extend(gen_symmetries(s2))

    print("00", validate_sols(gen_symmetries(s1)), gen_symmetries(s1))

    print("01", validate_sols(gen_symmetries(s2)), gen_symmetries(s2))

    s3 = np.roll(s2.copy(), 2)
    s4 = np.roll(s2.copy(), 2)

    print("1", validate_sols(gen_symmetries(s3.tolist())))

    sols.extend(gen_symmetries(s3.tolist()))

    print("2", validate_sols(gen_symmetries(s4.tolist(), [True, True, False, True, False, True, False])))

    sols.extend(gen_symmetries(s4.tolist(), [True, True, False, True, False, True, False]))



    s5 = np.roll(s1.copy(), -1)
    s5[[0,-1]] = [s5[-1], s5[0]]



    sym_s5 = gen_symmetries(s5.tolist(), [True, False, True, False, True, True, False])

    print("3", validate_sols(sym_s5))

    sols.extend(sym_s5)

    s6 = np.roll(sym_s5[2], -3)

    s6[[-2, -1]] = [s6[-1], s6[-2]]

    print("4", validate_sols(gen_symmetries(s6.tolist())))

    sols.extend(gen_symmetries(s6.tolist()))


    return sols


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


def validate_sols(sols):
    r = []
    for q in sols:
        r.append(validate(q))
    return r

for nn in range(7, 30):
    if nn % 2 == 1:
        print(nn)
        ss = queen_solver(nn)

print(len(ss))

# [0, 2, 4, 6, 1, 3, 5] cero, even and later odd
# [2, 4, 6, 1, 3, 5, 0] even, odd and later cero




