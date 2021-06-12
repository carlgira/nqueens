import os

def avg(*args):
    print(args)
    return sum(args)/(len(args))


def asd(a, b):
    return sum([1 for i,j in zip(a,b) if i>j]), sum([1 for i,j in zip(a,b) if i<j])


def diagonalDifference(arr):
    d1 = 0
    d2 = 0
    for i in range(len(arr)):
            d1 += arr[i][i]
            d2 += arr[i][len(arr)-1-i]
    return abs(d1-d2)

a = diagonalDifference([ [11, 2, 4],
                     [4, 5, 6],
                     [10, 8, -12]])

l = []

import datetime
date_time_str = '07:05:45PM'
date_time_obj = datetime.datetime.strptime(date_time_str, '%I:%M:%S%p')
print(date_time_obj.strftime("%H:%M:%S"))

def pSequences(n, p, l=[]):
    if len(l) == n:
        return 1
    total = 0
    for i in range(1, p+1):
        if len(l) == 0 or i*l[-1] <= p:
            lc = l[:]
            lc.append(i)
            total += pSequences(n, p, lc)
        elif i*l[-1] > p:
            return total
    return total

def oo(n, p):
    t = {}
    for i in range(1, p+1):
        t[i] = math.floor(p/i)
    c = copy.deepcopy(t)

    #print(c)
    for _ in range(n-2):
        tt = copy.deepcopy(c)
        #print(c)
        for j in range(1, p+1):
            if t[j] == 1:
                c[j] = tt[1]
            else:
                #print(sum([v for i, v in enumerate(tt.values())][:t[j]] ) , [v for i, v in enumerate(tt.values())][:t[j]] )
                c[j] += sum([v for i, v in enumerate(tt.values())][:t[j]] ) - tt[j]
    #print(c)
    return sum(list(c.values())) % (pow(10, 9) + 7)

import math
import copy
import random
def o(n, p):
    t = {}
    m = 0
    s = 0
    for i in range(1, p+1):
        t[i] = math.floor(p/i)
        if t[i] == 1:
            m = i
            t[i] = p - i + 1
            s = p - i + 1
            break
    c = copy.deepcopy(t)
    for k in range(n-2):
        tt = copy.deepcopy(c)
        print(c)
        for j in range(1, p - t[m]+1):
            c[j] += sum([v for i, v in enumerate(tt.values())][:t[j]] ) - tt[j]
        c[m] = s*tt[1]

    print(c)
    return sum(list(c.values())) % (pow(10, 9) + 7)


from collections import Counter
def oi(n, p):
    m = p
    d = [m]
    c = [1]
    counter = 0
    for i in range(2, p+1):
        v = math.floor(p/i)
        if m != v:
            counter += 1
            d.append(v)
            c.append(counter)
            m = v
            counter = 0
        else:
            c[-1] = c[-1] + 1

    total = [i*j for i, j in zip(c, d)]

    for k in range(n-2):
        tt = total[:]
        for k in range(len(c)):
            total[k] = sum([i for i in tt][:len(c) - k])*c[k]

    return sum(total) % (pow(10, 9) + 7)

def on(n, p):
    m = p
    d = [m]
    c = [1]
    counter = 0
    for i in range(2, p+1):
        v = math.floor(p/i)
        if m != v:
            counter += 1
            d.append(v)
            c.append(counter)
            m = v
            counter = 0
        else:
            c[-1] = c[-1] + 1

    total = [i*j for i, j in zip(c, d)]

    print(total)

    for k in range(n-2):
        tt = total[:]
        total[-1] = tt[0]*c[-1]
        prev = total[0]
        for k in range(len(c)-2, 0, -1):
            p = len(c) - k-1
            total[k] = (prev + tt[p])*c[k]
            prev += + tt[p]
        total[0] = (prev + tt[-1])
    return sum(total) % (pow(10, 9) + 7)
    #


# 50, 4 => 157019591

#for n in range(2, 5):
#    for p in range(3, 10):
#        print(n,p)
#        print(pSequences(n,p), o(n,p), oi(n, p))

#99 - 76 -> 14229301

'''
n 96 88 263550936 151279097.0
n 84 100 509277690 172939213.0
n 90 37 291802046 105587206.0
n 94 90 839381657 335523135.0
n 16 95 225317178 225315370.0
n 86 85 516429760 733200760.0
'''
# 1000 1000000000 => 336011589
#print(on(1000, 1000000000))
#print('asd', oo(100, 1), on(100, 1))

'''
c = 0
for _ in range(100):
    n = random.randint(2, 100)
    p = random.randint(2, 100)
    l1 = oo(n, p)
    l2 = on(n, p)

    if l1 != l2:
        c+= 1
        print('n', n, p, l1, l2)

print('nnnn' , c)
'''

#print(oo(5, 15), on(5, 15))
#print(oo(5, 8), on(5, 8))
#print(oo(14, 14), on(14, 14))
#print(oo(3, 8), on(3, 8))


# 1000, 1000000 => 430492191
#print(on(1000, 1000000))


import numpy as np

def iterateIt(a):
    z = a[:]
    rep = 0
    a = list(set(a))
    ml = len(a)
    while len(a) > 0:
        b = []
        for x in a:
            for y in a:
                if x != y:
                    b.append(abs(x-y))
        a = b
        a = list(set(a))
        if len(a) > ml:
            ml = len(a)
        rep +=1
        if len(a) > 1000 or rep > 100:
            print('bad', z, rep, len(a))
            break
        #elif len(a) == 0:
            #print('good', z)

    return rep

def iterateIti(a):
    rep = 0
    a = list(set(a))
    while len(a) > 0:
        b = []
        for x in a:
            for y in a:
                if x != y:
                    b.append(abs(x-y))
        a = b
        a = list(set(a))
        a.sort()
        rep += 1
        print(a)

        if len(a) == 0:
            return rep

        if len(a) == 1:
            return rep + 1

        m = a[1] - a[0]
        flag = True
        for i in range(1, len(a)-1):
            if a[i+1] - a[i] != m:
                flag = False
                break
        if flag:
            return rep + len(a)
    return rep




'''
for i in range(10000):
    #s = random.randint(4, 6)
    l = np.random.randint(100, size=5).tolist()
    m1 = max(l)
    m2 = min(l)
    r = iterateIti(l)
    print(l, r, m1-m2, m1, m2)

[17, 30, 55, 2, 99] 85 97 99 2
[81, 25, 43, 21, 93] 35 72 93 21
[20, 6, 57, 54, 88] 77 82 88 6
[6, 53, 96, 0, 7] 97 96 96 0
'''

#l = [i*10 for i in range(10, 1, -1)]
#print('c', iterateIt(l), iterateIti(l))

'''


'''


# 17 33 65 129 257 513 1025 2049 4097 8193 16385 32769 => 2048
# 49999 50000 => 2
# 1 291 48888 49176 => 48880


#l = [17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769]
#l2 = [1, 291, 48888, 49176]
#r = iterateIti(l2)
#print(r)


def kangaroo(x1, v1, x2, v2):
    if x1 == x2:
        return 'YES'
    elif v1 == v2:
        return 'NO'

    x = (x2 - x1)/(v1 - v2)
    y = v1*x + x1

    print(x, y)

    if x == int(x) and y == int(y) and x >= 0 and y >= 0:
        return 'YES'
    else:
        return 'NO'


def getTotalX(a, b):
    a.sort()
    b.sort()
    m = b[0]

    f1 = []
    for v in range(1, m+1):
        am = []
        for av in a:
            if v % av == 0:
                am.append(v)
        if len(am) == len(a):
            f1.append(am[0])

    f2 = []
    for v in f1:
        bm = []
        for bv in b:
            if bv % v == 0:
                bm.append(v)
        if len(bm) == len(b):
            f2.append(bm[0])
    return len(f2)

#print(getTotalX([1], [100]))


import itertools
import math
def getSquares(n):
    squares = list(itertools.permutations(range(1, n*n+1), n*n))
    mn = n*(n*n +1)/2
    magic_squares = []
    for s in squares:
        flag = True
        for r in range(n):
            if sum(s[r*n:r*n+n]) != mn or sum(s[r::n]) != mn:
                flag = False
                break
        if sum(s[::n+1]) != mn or sum(s[n-1::n-1][:n]) != mn:
            flag = False

        if flag:
            magic_squares.append(s)
    return magic_squares

def formingMagicSquare(s):
    mqs = getSquares(len(s))
    sf = []
    for m in s:
        sf.extend(m)
    mins = math.inf
    for mq in mqs:
        total = 0
        for i,j in zip(mq, sf):
            total += abs(i-j)
        if mins > total:
            mins = total

    return mins


#print(formingMagicSquare([[4, 9, 2], [3, 5, 7], [8, 1, 5]]))

def pickingNumbers(a):
    d = dict.fromkeys(a, 0)
    for v in a:
        d[v] = d[v] + 1

    if len(d.keys()) == 1:
        return len(d.values())

    m = -1
    for k in d.keys():

        if d[k] > m:
            m = d[k]

        if k-1 in d and (d[k] + d[k-1]) > m:
            m = d[k] + d[k-1]

        if k+1 in d and (d[k] + d[k+1]) > m:
            m = d[k] + d[k+1]

    return m

#l = [4 ,97 ,5 ,97 ,97 ,4 ,97 ,4 ,97 ,97 ,97 ,97 ,4 ,4 ,5 ,5 ,97 ,5 ,97 ,99 ,4 ,97 ,5 ,97 ,97 ,97 ,5 ,5 ,97 ,4 ,5 ,97 ,97 ,5 ,97 ,4 ,97 ,5 ,4 ,4 ,97 ,5 ,5 ,5 ,4 ,97 ,97 ,4 ,97 ,5 ,4 ,4 ,97 ,97 ,97 ,5 ,5 ,97 ,4 ,97 ,97 ,5 ,4 ,97 ,97 ,4 ,97 ,97 ,97 ,5 ,4 ,4 ,97 ,4 ,4 ,97 ,5 ,97 ,97 ,97 ,97 ,4 ,97 ,5 ,97 ,5 ,4 ,97 ,4 ,5 ,97 ,97 ,5 ,97 ,5 ,97 ,5 ,97 ,97 ,97]
#print(pickingNumbers(l))

import math
def transform(mt, nt, x0, y0, xt, yt, r):
    rt = mt*2 + nt*2
    r = r % rt

    if 0 <= r < nt:
        return xt, yt + r

    if nt <= r < nt + mt:
        return xt + (r % nt), yt

    if mt + nt <= r < nt*2 + mt:
        return xt, yt - (r % (mt + nt))

    if nt*2 + mt <= r < nt*2 + mt*2:
        return xt - (r % (nt*2 + mt)), yt


def matrixRotation1(matrix, r):
    m = len(matrix)
    n = len(matrix[0])
    result = matrix[:]
    for i in range(math.floor(m/2)):
        x0, y0 = i, i
        mt = m - x0*2
        nt = n - y0*2
        rt = mt*2 + nt*2
        nr = r % rt
        for t in range(n-1):
            xt, yt = transform(mt, nt, x0, y0, x0, t, r)
            result[xt][yt] = matrix[x0][t]

            xt, yt = transform(mt, nt, x0, y0, x0 + mt - 1, t, r)
            result[xt][yt] = matrix[x0 + mt - 1][t]

        for t in range(m-1):
            xt, yt = transform(mt, nt, x0, y0, t, y0, r)
            result[xt][yt] = matrix[t][y0]

            xt, yt = transform(mt, nt, x0, y0, t, y0 + nt - 1, r)
            print(xt, yt, t, y0 + nt - 1)
            result[xt][yt] = matrix[t][y0 + nt - 1]

    for _ in result:
        print( ' '.join(result))



from collections import deque

def matrixRotation(matrix, r):
    n = len(matrix)
    m = len(matrix[0])
    result = matrix[:]
    print(m)

    for i in range(min(math.floor(n/2), math.floor(m/2))):
        x0, y0 = i, i
        mt = m - x0*2
        nt = n - y0*2
        rt = mt*2 + nt*2 - 4
        nr = (r % rt)
        print('r', i, rt, nr)
        c1 = [row[x0] for row in matrix][y0:nt+y0]
        c2 = [row[x0 + mt - 1] for row in matrix][y0:nt+y0]

        print('c1', c1)
        print('c2', c2)
        print('c3', matrix[y0+nt-1][x0:x0+mt])
        print('c4', matrix[y0][x0:x0+mt])

        l = deque(c1[:-1] + matrix[y0+nt-1][x0:x0+mt][:-1] + list(reversed(c2))[:-1] + list(reversed(matrix[y0][x0:x0+mt]))[:-1])
        l.rotate(nr)
        l = list(l)

        c1r = l[0:nt]
        c2r = l[nt-1:nt-1+mt]
        c3r = l[nt-2+mt:nt*2-2+mt]
        c4r = l[-mt+1:]

        result[y0][x0:x0+mt] = [c1r[0]] + list(reversed(c4r))

        for i, v in enumerate(reversed(c3r)):
            result[y0 + i][x0 + mt-1] = v

        result[y0+nt-1][x0:x0+mt] = c2r

        for i, v in enumerate(c1r):
            result[y0 + i][x0] = v

    for re in result:
        print(' '.join(map(str, re)))


l1 = [[1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16]]

l2 = [[1, 2, 3, 4],
      [7, 8, 9, 10],
      [13, 14, 15, 16],
      [19, 20, 21, 22],
      [25, 26, 27, 28]]

#l3 = np.random.randint(100, size=(10,8)).tolist()
l3 = [[36, 20, 25, 48, 10, 47, 27, 69],
      [32, 18, 96, 36, 27, 29, 72, 28],
      [9, 27, 30, 57, 47, 30, 68, 56],
      [25, 91, 6, 37, 15, 76, 47, 27],
      [99, 51, 0, 4, 24, 53, 79, 48],
      [96, 30, 0, 34, 33, 36, 9, 39],
      [38, 59, 29, 64, 87, 25, 92, 89],
      [19, 25, 40, 32, 59, 74, 42, 8],
      [25, 13, 68, 54, 71, 63, 79, 63],
      [61, 56, 90, 16, 39, 75, 95, 44]]

l4 = [[1,2,3,4]]

#matrixRotation(l5, 20)


#import  numpy as np
#ar = np.random.random_integers(10, size=(5,5)).tolist()

#print(ar)
#matrixRotation(ar, 1)


def getBattery(events):
    batery = 50
    for e in events:
        batery = (batery + e)
        if batery < 0:
            batery = 0
        if batery > 100:
            batery = 100
    return batery

#print(getBattery([4, 25, -30, 70, -10]))


def get_dict(one):
    dict_one = dict.fromkeys(one, 0)
    for l in one:
        dict_one[l] = dict_one[l] + 1
    return dict_one

def is_anagram(dict_one, dict_two):
    if len(dict_one.keys()) != len(dict_two.keys()):
        return False
    for k in dict_one.keys():
        if not (k in dict_two and dict_one[k] == dict_two[k]):
            return False
    return True

def stringAnagram(dictionary, query):
    result = []
    d_dictionary = [get_dict(l) for l in dictionary]
    d_query = [get_dict(l) for l in query]

    for q in d_query:
        counter = 0
        for d in d_dictionary:
            if is_anagram(q, d):
                counter += 1
        result.append(counter)
    return result


#print(stringAnagram(['heater', 'cold', 'clod', 'reheat', 'docl'], ['codl', 'heater', 'abcd']))


def is_ancestor(children, tree, l=[]):
    if len(tree.keys()) == 0:
        return False

    if children in tree:
        return True

    for k in tree.keys():
        if k not in l:
            l.append(k)
            r = is_ancestor(children, tree[k], l[:])
            if r:
                return True

    return False

def similarPair1(n, k, edges):
    tree = {}
    for i in range(1, n+1):
        tree[i] = {}

    for edge in edges:
        parent, children = edge
        tree[parent][children] = tree[children]

    counter = 0

    for n1 in range(1, n+1):
        for n2 in range(1, n+1):
            if n1 == n2:
                continue
            if abs(n1-n2) <= k:

                ancestor = is_ancestor(n2, tree[n1])
                print(n1, n2, ancestor)
                if ancestor:
                    counter += 1

    return counter

#tree = np.random.randint(1, 100, size=(100,2)).tolist()

#tree = [[27, 45], [46, 92], [52, 86], [84, 36], [94, 51], [25, 88], [43, 3], [7, 50], [95, 60], [8, 22], [23, 30], [87, 93], [80, 95], [42, 10], [94, 19], [53, 1], [25, 19], [42, 86], [78, 94], [14, 11], [54, 69], [30, 24], [47, 94], [13, 32], [97, 58], [3, 80], [19, 93], [98, 54], [56, 55], [30, 45], [14, 42], [55, 44], [75, 36], [89, 57], [73, 68], [13, 6], [30, 95], [92, 63], [5, 34], [87, 31], [41, 75], [17, 37], [75, 3], [57, 86], [64, 31], [76, 50], [60, 66], [66, 45], [23, 49], [99, 81], [65, 20], [28, 26], [21, 52], [30, 56], [27, 54], [47, 1], [24, 16], [73, 63], [41, 45], [28, 56], [92, 95], [46, 90], [13, 54], [74, 4], [57, 10], [70, 27], [14, 63], [48, 64], [47, 31], [51, 47], [79, 13], [69, 92], [67, 73], [25, 9], [62, 37], [16, 4], [95, 99], [50, 45], [64, 18], [72, 58], [82, 79], [7, 53], [28, 99], [50, 15], [17, 75], [47, 12], [1, 80], [74, 89], [81, 98], [9, 12], [6, 91], [26, 91], [9, 60], [77, 51], [93, 38], [76, 86], [58, 9], [54, 83], [59, 14], [65, 33]]



import bisect

mem = {}
def calc(k, tree, ancestors, counter=0):

    if len(tree.keys()) == 0:
        return 0

    global mem

    for t in tree.keys():

        left = bisect.bisect_left(ancestors, t-k)
        right = bisect.bisect_right(ancestors, k+t)

        if left != len(ancestors) or right != 0:
            counter += right-left
        new_ancestors = ancestors[:]
        bisect.insort(new_ancestors, t)
        counter += calc(k, tree[t], new_ancestors)

    return counter


def similarPair(n, k, edges):
    tree = {}
    for i in range(1, n+1):
        tree[i] = {}

    for edge in edges:
        parent, children = edge
        tree[parent][children] = tree[children]

    root = edges[0][0]
    return calc(k, tree[root], [root])

from simi import ll
#l1 =[[3,2], [3,1], [1,4], [1,5]]
#print(similarPair(5, 2, l1))

#l2 = [[1, 2], [1, 3], [3, 4], [3, 5], [3, 6]]
#print(similarPair(6, 2, l2))

#print(similarPair(99999, 90765, ll))



# 4058469201

import bisect
def insertionSort(arr):
    result = []
    counter = 0
    for i, a in enumerate(arr):
        ip = bisect.bisect(result, a)
        counter += abs(ip - i)
        bisect.insort(result, a)

    return counter

#print(insertionSort([1, 1, 1, 2, 2 ])) # => 0
#print(insertionSort([2, 1, 3, 1, 2]))

def getSum( BITree, index):
    sum = 0
    while (index > 0):
        sum += BITree[index]
        index -= index & (-index)
    return sum

def updateBIT(BITree, n, index, val):
    while (index <= n):
        BITree[index] += val
        index += index & (-index)

def minimumBribes(q):
    n = len(q)
    invcount = 0
    maxElement = max(q)
    BIT = [0] * (maxElement + 1)
    for i in range(n - 1, -1, -1):

        invcount += getSum(BIT, q[i] - 1)
        updateBIT(BIT, maxElement, q[i], 1)
    return invcount

def minimumSwaps(arr):
    result = 0
    i = 0
    while i < len(arr):
        if arr[i] != i+1:
            arr[arr[i]-1], arr[i],  =  arr[i], arr[arr[i]-1]
            result += 1
        else:
            i += 1
    return result

#print(minimumSwaps([7, 1, 3, 2, 4, 5, 6] ))

from array import lll

def arrayManipulation1(n, queries):
    maxElement = n
    BIT = [0] * (maxElement + 1)
    for query in queries:
        a, b, k = query
        updateBIT(BIT, maxElement, a, k)
        updateBIT(BIT, maxElement, b+1, -k)
    r = [getSum(BIT, i) for i in range(n)]
    return max(r)


def arrayManipulation(n, queries):
    result = [0]*(n+1)
    for query in queries:
        a, b, k = query
        result[a-1] = result[a-1] + k
        result[b] = result[b] - k

    max_value = -math.inf
    for i in range(1, n):
        result[i] = result[i] + result[i-1]
        if max_value < result[i]:
            max_value = result[i]

    return max_value


#print(arrayManipulation(10000000, lll))

def countTriplets(arr, r):
    result = 0
    arr_sorted = sorted(arr)
    arr_dict = Counter(arr_sorted)

    for i, v in enumerate(set(arr_sorted)):
        if v*r in arr_dict and v*r*r in arr_dict:
            v1 = arr_dict[v]
            v2 = arr_dict[v*r]
            v3 = arr_dict[v*r*r]
            if v*r == v and v == v*r*r:
                v2 -= (i+2)
                v3 -= (i+3)
            result += v1*v2*v3

    return result


#print(countTriplets([1]*100, 1)) #161700

def prepare_crosswordPuzzle(crossword, words, s):
    words = words.split(";")
    rows = {}
    cols = {}
    c_rows = {}
    c_cols = {}
    for i in range(len(crossword)):
        rows[i] = []
        cols[i] = []
        c_rows[i] = []
        c_cols[i] = []

    crossword_col = ['' for _ in range(len(crossword[0]))]
    for ri, row in enumerate(crossword):
        for ci, col in enumerate(row):
            crossword_col[ci] = crossword_col[ci] + col

    for ri, row in enumerate(crossword):
        index = 0
        if ri == 5:
            index = index
        while index < len(row):
            value = row[index:]
            if '-' in value:
                begin = value.index('-')
                end = begin+1
                if s in value[begin:]:
                    end = begin + value[begin:].index(s)
                else:
                    end = len(value)

                if (end - begin) != 1:
                    rows[ri].append((index + begin, index + end))

                if '-' in value[end+1:]:
                    index += end
                else:
                    break
            else:
                break

    for ci, col in enumerate(crossword_col):
        index = 0
        while index < len(col):
            value = col[index:]
            if '-' in value:
                begin = value.index('-')
                end = begin+1
                if s in value[begin:]:
                    end = begin + value[begin:].index(s)
                else:
                    end = len(value)

                if (end - begin) != 1:
                    cols[ci].append((index + begin, index + end))

                if '-' in value[end+1:]:
                    index += end
                else:
                    break
            else:
                break

    for rk in rows.keys():
        for ck in cols.keys():
                for ci in cols[ck]:
                    if len(rows[rk]) > 0:
                        inter = set([rk]).intersection(set(range(ci[0], ci[1])))
                        if len(inter) != 0:
                            c_cols[ck].append((ck, rk))
                            c_rows[rk].append((rk, ck))

    cross = {}
    for rk in rows.keys():
        if len(rows[rk]) > 0:
            for i, w in enumerate(rows[rk]):
                cross[('r', rk, i)] = [word for word in words if len(word) == w[1]-w[0]]

    for ck in cols.keys():
        if len(cols[ck]) > 0:
            for i, w in enumerate(cols[ck]):
                cross[('c', ck, i)] = [word for word in words if len(word) == w[1]-w[0]]

    return rows, cols, c_rows, c_cols, crossword_col, cross

def check_set(rows, cols, c_rows, c_cols, crossword_rows, crossword_cols, tp, w):
    t, k, p = tp
    if t == 'r':
        opts = rows[k][p]
        value = crossword_rows[k][opts[0]:opts[1]]
        if len(value.replace('-', '')) == 0 and len(value) == len(w):
            return True
        else:
            for c in c_rows[k]:
                x,y = c
                if crossword_rows[x][y] != '-' and crossword_rows[x][y] != w[y - opts[0]]:
                    return False

    if t == 'c':
        opts = cols[k][p]
        value = crossword_cols[k][opts[0]:opts[1]]
        if len(value.replace('-', '')) == 0 and len(value) == len(w):
            return True
        else:
            for c in c_cols[k]:
                x,y = c
                if crossword_cols[x][y] != '-' and crossword_cols[x][y] != w[y - opts[0]]:
                    return False

    return True

def set_word(rows, cols, crossword_rows, crossword_cols, tp, w):
    t, k, p = tp
    if t == 'r':
        opts = rows[k][p]
        crossword_rows[k] = crossword_rows[k].replace(crossword_rows[k][opts[0]:opts[1]], w)

        for i in range(opts[0], opts[1]):
            temp = list(crossword_cols[i])
            temp[k] = crossword_rows[k][i]
            crossword_cols[i] = ''.join(temp)

    if t == 'c':
        opts = cols[k][p]
        crossword_cols[k] = crossword_cols[k].replace(crossword_cols[k][opts[0]:opts[1]], w)

        for i in range(opts[0], opts[1]):
            temp = list(crossword_rows[i])
            temp[k] = crossword_cols[k][i]
            crossword_rows[i] = ''.join(temp)


sol = None
def crossword_puzzle_find(rows, cols, c_rows, c_cols, crossword_rows, crossword_cols, cross, k=0, q=[]):

    if k not in rows or len(list(cross.keys())) <= k:
        global sol
        sol = crossword_rows
        return True

    tp = list(cross.keys())[k]

    for w in cross[tp]:
        if w not in q and check_set(rows, cols, c_rows, c_cols, crossword_rows, crossword_cols, tp, w):
            cq = q[:]
            cq.append(w)
            crossword_rows_c = crossword_rows[:]
            crossword_cols_c = crossword_cols[:]
            set_word(rows, cols, crossword_rows_c, crossword_cols_c, tp, w)

            flag = crossword_puzzle_find(rows, cols, c_rows, c_cols, crossword_rows_c, crossword_cols_c, cross, k+1, cq)
            if flag:
                return crossword_rows_c

    return False

def crosswordPuzzle(crossword, words):

    s = '+'
    if 'X' in ''.join(crossword):
        s = 'X'

    rows, cols, c_rows, c_cols, crossword_cols, cross = prepare_crosswordPuzzle(crossword, words, s)



    for k in list(cross.keys()):
        if len(cross[k]) == 1:
            set_word(rows, cols, crossword, crossword_cols, k, cross[k][0])
            del cross[k]

    crossword_puzzle_find(rows, cols, c_rows, c_cols, crossword, crossword_cols, cross)

    return sol


c = ['+-++++++++',
     '+-++++++++',
     '+-++++++++',
     '+-----++++',
     '+-+++-++++',
     '+-+++-++++',
     '+++++-++++',
     '++------++',
     '+++++-++++',
     '+++++-++++']
w = 'LONDON;DELHI;ICELAND;ANKARA'

#print(crosswordPuzzle(c, w))


c1 =['+-++++++++',
     '+-++++++++',
     '+-------++',
     '+-++++++++',
     '+-++++++++',
     '+------+++',
     '+-+++-++++',
     '+++++-++++',
     '+++++-++++',
     '++++++++++']
w1 = 'AGRA;NORWAY;ENGLAND;GWALIOR'

#print(crosswordPuzzle(c1, w1))

'''
    +E++++++++

    +N++++++++

    +GWALIOR++

    +L++++++++

    +A++++++++

    +NORWAY+++

    +D+++G++++

    +++++R++++

    +++++A++++

    ++++++++++
'''

c2 = ['XXXXXX-XXX',
      'XX------XX',
      'XXXXXX-XXX',
      'XXXXXX-XXX',
      'XXX------X',
      'XXXXXX-X-X',
      'XXXXXX-X-X',
      'XXXXXXXX-X',
      'XXXXXXXX-X',
      'XXXXXXXX-X']
w2 = 'ICELAND;MEXICO;PANAMA;ALMATY'

#print(crosswordPuzzle(c2, w2))

'''
    XXXXXXIXXX

    XXMEXICOXX

    XXXXXXEXXX

    XXXXXXLXXX

    XXXPANAMAX

    XXXXXXNXLX

    XXXXXXDXMX

    XXXXXXXXAX

    XXXXXXXXTX

    XXXXXXXXYX
'''

c4 = ['+-++++++++',
      '+-++-+++++',
      '+-------++',
      '+-++-+++++',
      '+-++-+++++',
      '+-++-+++++',
      '++++-+++++',
      '++++-+++++',
      '++++++++++',
      '----------']
w4 = 'CALIFORNIA;NIGERIA;CANADA;TELAVIV'

#print(crosswordPuzzle(c4, w4))

c5 = ['+-++++++++',
      '+-------++',
      '+-++-+++++',
      '+-------++',
      '+-++-++++-',
      '+-++-++++-',
      '+-++------',
      '+++++++++-',
      '++++++++++',
      '++++++++++']

w5 = 'ANDAMAN;MANIPUR;ICELAND;ALLEPY;YANGON;PUNE'

#print(crosswordPuzzle(c5, w5))
