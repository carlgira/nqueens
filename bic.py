import pandas as pd
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_json('bici.json', orient='records')

# Add 1G to remove gravity effect on data
data.loc[: , 'x'] = data.x + 1000

# Normalize values to be in meters by second
data.loc[: , 'x'] = data.x/100
data.loc[: , 'y'] = data.y/100
data.loc[: , 'z'] = data.z/100

# Batch acceleration constants
batch_size = 10
samples_by_second = 10

# Time variable
t = np.arange(0.0, 1.0/(samples_by_second/batch_size), 1/samples_by_second)

# Simulate data arrival, acceleration in batches
last_values = []
max_queue_size = 3
for i in list(range(0,  len(data) - len(data)%batch_size, batch_size)):
    batch = data[i : i + batch_size]

    fx = np.fft.fft(batch.x)
    freq = np.fft.fftfreq(t.shape[-1])

    dx = batch.x/np.sin(2*np.pi*(1/samples_by_second)*(samples_by_second/batch_size))
    vdx = np.pi*dx*np.cos(2*np.pi*(1/samples_by_second)*(samples_by_second/batch_size))

    dy = batch.y/np.sin(2*np.pi*(1/samples_by_second)*(samples_by_second/batch_size))
    vdy = np.pi*dy*np.cos(2*np.pi*(1/samples_by_second)*(samples_by_second/batch_size))

    dz = batch.z/np.sin(2*np.pi*(1/samples_by_second)*(samples_by_second/batch_size))
    vdz = np.pi*dz*np.cos(2*np.pi*(1/samples_by_second)*(samples_by_second/batch_size))


    vx = integrate.simps(batch.x, t)
    vy = integrate.simps(batch.y, t)
    vz = integrate.simps(batch.z, t)

    v = np.sqrt( np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2))
    vks = v*3600/1000
    #print(vks)

    vd = np.sqrt( np.power(vdx, 2) + np.power(vdy, 2) + np.power(vdz, 2))
    vdks = vd*360/1000

    last_values.append(np.mean(vdks))
    last_values = last_values[-max_queue_size:]

    vmx = (np.abs(np.max(batch.x)) - np.abs(np.min(batch.x)))*( np.abs(np.argmax(batch.x) - np.argmin(batch.x))*1/samples_by_second)
    vmy = (np.abs(np.max(batch.y)) - np.abs(np.min(batch.y)))*( np.abs(np.argmax(batch.y) - np.argmin(batch.y))*1/samples_by_second)
    #vmz = (np.abs(np.max(batch.z)) - np.abs(np.min(batch.z)))*( np.abs(np.argmax(batch.z) - np.argmin(batch.z))*1/samples_by_second)

    vm = np.sqrt( np.power(vmx, 2) + np.power(vmy, 2))  # + np.power(vmz, 2))
    vmks = vm*3600/1000

    print(np.mean(last_values), vks, vmks)

