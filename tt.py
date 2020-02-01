



def solutiont1(H):
    t = sorted([[i,H[i]] for i in range(len(H))], key=lambda x: x[1])
    print(t)

    counter = 0
    for v in t:
        if v[1] == 0:
            continue
        counter += 1
        t[v[0]][1] = 0
        for l in range(v[0], 0, -1):
            if v[1] <= t[l][1]:
                t[l][1] = t[l][1] - v[1]
            else:
                break
        for r in range(v[0], len(t)):
            if v[1] <= t[r][1]:
                t[r][1] = t[r][1] - v[1]
            else:
                break

        print(t)
    return counter


#H = [8, 8, 5, 7, 9, 8, 7, 4, 8]
#print(solution(H))


def solutiont2(A):
    b = [0, 1, 1, 2, 1, 2, 2, 3]
    return sum([b[v] for v in A])


import itertools
import random


A = [1, 2, 4, 5, 7, 29, 30]
R = [2, 2, 2, 2, 2]

def create_sol(values):
    count = sum([v if v == 7 else 1 for v in values])
    cost = sum(values)
    return {'value': values, 'cost': cost, 'count': count}

def new_solution(A):
    for _ in range(20):
        values = [7]*random.randint(0, 3)
        values.extend([2]*random.randint(0, int(13/(len(values)+1))))
        random.shuffle(values)

        p = create_sol(values)

        if valid_sol(p['value'], A) and len(A) <= p['count'] and len(values) > 0:
            return p

    return None

def valid_sol(sol, A):
    if len(A) == 0:
        return True

    if len(sol) == 0:
        return False

    end = A[0] + (sol[0] if sol[0] == 7 else 1)
    r = [v for v in A if v >= end]
    return valid_sol(sol[1:], r)


def mutate_solution(sol, A):
    new_sol = sol.copy()
    r = random.randint(0, 9)

    if r < 3:
        e = random.randint(0, len(new_sol))
        new_sol = [new_sol[i] for i in range(len(new_sol)) if i != e]
    elif r < 6:
        v = [2,7][random.randint(0, 1)]
        p = random.randint(0, len(new_sol)-1)
        new_sol.insert(p, v)
    elif r < 9:
        v = [2,7][random.randint(0, 1)]
        p = random.randint(0, len(new_sol)-1)
        new_sol[p] = v

    if valid_sol(new_sol, A):
        return create_sol(new_sol)

    return None


def solution4(A):

    if len(A) > 25:
        return 25

    population_size = 100
    epochs = 1000
    P = []

    # Create population
    for _ in range(population_size):
        n = new_solution(A)
        if n is not None:
            P.append(n)

    # Order Population
    P = sorted(P, key=lambda k: k['cost'])

    # Iterate
    for it in range(epochs):
        for sol in P:


            # Mutate
            if random.random() < 0.1:
                sol = mutate_solution(sol['value'], A)
                if sol is not None:
                    P.append(sol)

            # Add new Solutions
            if random.random() < 0.1:
                sol = new_solution(A)
                if sol is not None:
                    P.append(sol)

        # Order and resize population
        P = sorted(P, key=lambda k: k['cost'])
        P = P[0:population_size]

    if len(P) == 0 or P[0]['cost'] >= 25:
        return 25

    return P[0]['cost']


#print(solution([1, 2, 4, 5, 7, 29, 30]))

'''
solution([1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 29])



for _ in range(100):
    v = random.sample(range(1, 30), random.randint(1, 30))
    v = sorted(v)

    print(v)
    solution(v)

'''
from operator import itemgetter


def calculate_path(node, C, cameras, sol, best_sol):
    sol.append(node)
    flag = True
    for v in C[node]:
        c = v[1]
        n = v[0]

        if n in sol or c in cameras:
            continue
        else:
            flag = False
            calculate_path(n, C, cameras, sol.copy(), best_sol)

    if flag and len(sol) > len(best_sol):
        best_sol.clear()
        best_sol.extend(sol)

def build_matrix(A, B):
    C = [[] for _ in range(len(A)+1)]
    for a, b, i in zip(A,B, range(len(A))):
        C[a].append((b,i))
        C[b].append((a,i))
    return C


def calculate_longest_path(C, cameras):
    best_sol = []
    [calculate_path(i, C, cameras, [], best_sol) for i in range(len(C))]
    return len(best_sol)


def solutiont5(A, B, K):
    sols = []
    C = build_matrix(A,B)
    for _ in range(1000):

        cameras = random.sample(range(1, len(A)), K)
        r = calculate_longest_path(C, cameras)
        sols.append({'camera': cameras, 'count': r})

    sols = sorted(sols, key=lambda k: k['count'])

    print(sols)

    return sols[0]['count']-1

A = [5,1,0,2,7,0,6,6,1]
B = [1,0,7,4,2,6,8,3,9]
K = 2

'''
l = random.randint(30, 50)
K = random.randint(5, 30)

A = []
B = []

for _ in range(l):
    a = random.randint(0,l-1)
    b = random.randint(0,l-1)
    if a != b:
        A.append(a)
        B.append(b)

print(len(A), K, A, B)

count = solutiont5(A, B, K)

print("sol", count)
'''
def validate_square(node, A):
    L = min(node)+1
    N = len(A)
    M = len(A[0])
    X = node[0]
    Y = node[1]


    return (0 <= L <= min([N,M])) and (0 <= X <= N-L) and (0 <= Y <= M-L) and min([A[x][y] for x in range(L) for y in range(L)])


def longest_path(node, A, sol, best_sol):

    if A[node[0]][node[1]]:
        sol.append(node)

    if node[0] < len(A)-1 and A[node[0]+1][node[1]]:
        longest_path([node[0]+1, node[1]], A, sol.copy(), best_sol)

    if node[1] < len(A[0])-1 and A[node[0]][node[1]+1]:
        longest_path([node[0], node[1]+1], A, sol.copy(), best_sol)

    if node[0] == node[1] and A[node[0]][node[1]] and len(sol) > len(best_sol) and validate_square(node, A):
        print(sol)
        best_sol.clear()
        best_sol.extend(sol)


def solution(A):
    if A == [[True]]:
        return 1
    best_sol = []
    longest_path([0,0], A, [], best_sol)

    if len(best_sol) > 0:
        for l in range(best_sol[-1][0]+1, 1,-1):
            final = min([validate_square([x,y], A) for x in range(l) for y in range(l)])
            if final:
                return l
    return 0


A = [ [True, True, True, False],
      [True, True, True, False],
      [True, True, True, False],
      [True, True, True, True],
      [False, True, True, True],
      [True, False, True, True]]

print(solution(A))


A = [ [True, True, False, False],
      [True, False, False, False],
      [False, True, False, True]]


print(solution(A))

A = [[True]]

print(solution(A))

import numpy as np
A = [random.randint(0,1) == 0 for _ in range(300)]
A = np.array(A).reshape(30,10)





print(solution(A.tolist()))
