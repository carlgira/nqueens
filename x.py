from itertools import permutations
sol = [3,0,4,1,5,2]

def check(vec):
	n = len(vec)
	cols = range(len(vec))
	return (n == len(set(vec[i] + i for i in cols))
			== len(set(vec[i] - i for i in cols)))



for i in range(6, 10):

	if i % 2 == 1:
		sol.append(i)
	else:
		sol.insert(0,i)

	print("check", sol)

	if check(sol):
		print(sol)
	else:
		break




