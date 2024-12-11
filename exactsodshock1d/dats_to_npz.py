"""
Code I have written specifically for taking the data produced by the exact solver for the sod shock problem 
(which comes in the form of 41 .dat files, with one for each time step)
and writing it all to a compressed npz file with the same formatting as our other data.
"""
import numpy as np
import os

directory = '/Users/zaksaeed/Downloads/exact_riemann/' #The directory the .dat files are stored in on my (Zak's) computer
savename = 'exactsodshock'
# savename = 'exactsodshockx2'
# savename = 'exactsodshockx4'

time_start = 0.
time_end = 0.4
times_recorded = 41

files = sorted([file for file in os.listdir(directory) if file.endswith('.dat')])

x = np.loadtxt(directory + files[0], dtype=np.float64, skiprows=3, usecols=1)
t = np.linspace(time_start, time_end, times_recorded)

rho = np.zeros((t.size, x.size))
v = np.copy(rho)
p = np.copy(rho)

i = 0
for file in files:
    rho[i], p[i], v[i] = np.loadtxt(directory + file, dtype=np.float64, skiprows=3, usecols=(2, 3, 4), unpack=True)
    i += 1

np.savez_compressed(savename, t=t, x=x, rho=rho, v=v, p=p)