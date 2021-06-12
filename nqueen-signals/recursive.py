import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
import nqueens
import time

def convert_list(data, n=-1, set_size=-1):
    data = data.astype('uint8')
    if n != -1:
        num_bits = math.floor(math.log2(n-1)) + 1
        if set_size == -1:
            set_size = num_bits

    if num_bits*len(data) % set_size != 0:
        return np.array([])

    return np.reshape(np.packbits(np.flip(np.reshape(np.unpackbits(np.reshape(data, (-1, 1)), axis=1)[:, -num_bits:], (-1, set_size)), axis=1), bitorder='little', axis=-1), -1)


def decode_wave(values, n, num_of_values, separation):
    sol_values = np.abs(np.fft.irfft(values, num_of_values*separation)[::num_of_values])
    min_value = np.min(sol_values)
    max_value = np.max(sol_values)
    div_size = (max_value-min_value)/(n-1)
    return np.round(sol_values/div_size)



def validate(sol, n):

    if len(sol) != n:
        return False

    if np.sum(sol) != n*(n+1)/2:
        return False

    if len(sol) != len(set(sol)):
        return False

    v1 = [sol[i] + i for i in range(len(sol))]

    if len(v1) != len(set(v1)):
        return False

    v2 = [sol[i] - i for i in range(len(sol))]

    if len(v2) != len(set(v2)):
        return False

    return True

class RecursiveFunction:

    def __init__(self, center, times, dimensions):
        self.center_index = -1
        self.center = center
        self.times = times
        self.center_points = None
        self.dimensions = dimensions
        self.center_points = []
    '''
    def __init__(self, center_points, dimensions):
        self.center_index = 0
        self.center_points = center_points
        self.center = self.center_points[self.center_index]
        self.times = len(center_points)
        self.dimensions = dimensions
    '''

    @staticmethod
    def get_pop(pop):
        return RecursiveFunction(pop.center, pop.times, pop.dimensions)

    def next_center(self):
        if self.center_index != -1 and self.center_index < len(self.center_points)-1:
            self.center_index += 1
            self.center = self.center_points[self.center_index]

    def eval(self, init_point):
        result = [init_point]
        distance = lambda p1, p2: math.sqrt((p2.real - p1.real)**2 + (p2.imag - p1.imag)**2)
        distances = [distance(self.center, result[-1])]
        for step in range(self.times):
            for d in self.dimensions:
                result.append(result[-1]*d + self.center)

            distances.append(distance(self.center, result[-1]))
            self.next_center()

        return np.array(result[:-1])


#t = RecursiveFunction(center=0j, times=3, dimensions=[-1j, -1j, -1j, -0.5j])
#t = RecursiveFunction([0j, 0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j], dimensions=[-1j, -1j, -1j, -0.5j])

#r = t.eval(2+2j)
#print(r)



class GenPop:

    def __init__(self, n, conf):
        self.n = n
        self.center_max = conf['center_max']
        self.precision = conf['precision']
        self.precision_flag = conf['precision_flag']
        self.times_min = conf['times_min']
        self.times_max = conf['times_max']
        self.dimensions_min = self.n - 2
        self.dimensions_max = self.n*self.n
        self.bits_min = conf['bits_min']
        self.bits_max = self.n*2
        self.init_point_max = conf['init_point_max']
        self.max_separation = conf['max_separation']
        self.min_separation = conf['min_separation']

        self.times = round(random.uniform(self.times_min, self.times_max))
        self.separation = round(random.uniform(self.min_separation, self.max_separation))
        self.center = self.gen_complex(self.center_max, self.precision)
        self.num_dimensions = round(random.uniform(self.dimensions_min, self.dimensions_max))
        self.dimensions = np.random.random(self.num_dimensions) + np.random.random(self.num_dimensions) * 1j
        if self.precision_flag:
            self.dimensions = np.round(np.random.random(self.num_dimensions), self.precision) \
                          + np.round(np.random.random(self.num_dimensions), self.precision) * 1j
        self.bits = random.uniform(self.bits_min, self.bits_max)
        self.init_point = self.gen_complex(self.init_point_max, self.precision)
        self.fun = RecursiveFunction(center=self.center, times=self.times, dimensions=self.dimensions)
        self.score = 0

    def gen_complex(self, max_value, precision):
        if self.precision_flag:
            return round(random.uniform(-max_value, max_value), precision) \
               + round(random.uniform(-max_value, max_value), precision)*1j
        else:
            return random.uniform(-max_value, max_value)  + random.uniform(-max_value, max_value)*1j

    def gen_fun(self):
        return self.fun.eval(self.init_point)

    def calculate_score(self, sols):
        max_value = self.n * len(sols)*4
        num_of_sols = len(sols)
        num_of_values = num_of_sols*self.n
        values = self.gen_fun()
        real_values = decode_wave(values, self.n, num_of_values, self.separation)

        unique, counts = np.unique(real_values, return_counts=True)
        wave_dict = dict(zip(unique, counts))

        if len(real_values) <= num_of_values:
            self.score += len(real_values)
        else:
            self.score += num_of_values - len(real_values)

        for i in range(self.n):
            if i in wave_dict:
                if wave_dict[i] <= num_of_sols:
                    self.score += wave_dict[i]
                else:
                    self.score += num_of_sols - wave_dict[i]

        #for i in range(self.n, max(list(wave_dict.keys())) + 1):
        #    if i in wave_dict:
        #        self.score -= wave_dict[i]

        if len(real_values) > num_of_values:
            real_values = real_values[0:num_of_values]
        elif len(real_values) < num_of_values:
            real_values = np.pad(real_values, (0, num_of_values - len(real_values)))

        group_values = np.reshape(real_values, (-1, self.n))
        self.score += sum([len(set(g.tolist())) for g in group_values])

        self.score += len(np.unique(np.array([tuple(g) for g in group_values]), axis=0))
        self.score += sum([validate(g, self.n) for g in group_values]*self.n)


    def get_df(self):
        return {'times': self.times, 'center': self.center, 'separation': self.separation, 'num_dimensions': self.num_dimensions,
                'dimensions': self.dimensions, 'bits': self.bits, 'init_point': self.init_point,
                'score': self.score}


conf1 = {
    'center_max': 10,
    'precision': 3,
    'precision_flag': False,
    'times_min': 10,
    'times_max': 1500,
    'min_separation': 35,
    'max_separation': 50,
    'bits_min': 2,
    'init_point_max': 10000,
    'file': 'data_conf1.csv'
}


def evo(n):
    start_time = time.time()
    sols = nqueens.n_queens(n).all_solutions
    l = []
    iterations = 100000
    conf = conf1
    for _ in range(iterations):
        r = GenPop(n, conf)
        r.calculate_score(sols)
        d = r.get_df()
        l.append(d)

    df = pd.DataFrame(l)
    df.to_csv(conf['file'], mode='a', header=False, index=False)

    print("--- %s seconds, %s iterations ---" % ((time.time() - start_time), iterations))

# --- 13.96472692489624 seconds, 1000 iterations ---
# 2 min, 10000
# 20 min 100000
n = 5
#evo(n)


dim = [0.92492244+0.46193063j, 0.27945379+0.90593177j, 0.3414314 +0.00910187j, 0.03996786+0.43673661j,
       0.54821727+0.89330067j, 0.38737076+0.174801j, 0.26678142+0.98142178j, 0.71830902+0.60738465j,
       0.58904418+0.46654467j, 0.97509859+0.04341204j, 0.65999461+0.08500968j, 0.47337129+0.5345425j,
       0.49260009+0.84500811j, 0.59537076+0.44559715j, 0.35513893+0.99943419j]
q = RecursiveFunction(7.818657349404031+1.6863494329477557j, 57, dim)
v = q.eval(-463.36603920168636+7956.242823554341j)
vc = convert_list(np.fft.irfft(v), n)
cv = np.fft.irfft(v)
plt.plot(np.arange(len(cv)), cv)
plt.show()
print(cv)



'''
import pandas as pd
df = pd.read_csv(conf['file'])
df.columns = ['times', 'center', 'separation', 'num_dimensions', 'dimensions', 'bits', 'init_point', 'score']
df1 = df[df.score > 110]
df1.sort_values('score')
'''




