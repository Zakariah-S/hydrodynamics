from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

"""
Recommended: periodic boundary conditions. You can change this by editing the evolution.py file.
"""
def wave(x_start, x_end, t_final, nx, nt, savename = None):
    # Initialize independent and dependent variables

    x = np.linspace(x_start, x_end, nx)
    rho = np.zeros(nx)
    v = np.zeros(nx)
    P = np.zeros(nx)
    
    rho = 1. + 0.000001 * np.sin(2. * np.pi * x)
    P = 1. + 0.000001 * np.sin(2. * np.pi * x)

    U, t, x = initialize(x, t_final, nt, rho, v, P)
    U = evolve(U, t, x[1] - x[0], nx)
    if savename: save_data(savename, U, x, t)

if __name__ == '__main__':
    wave(x_start=0.,       #left side of tube
            x_end = 1.,       #right side of tube
            t_final = .85,    #time we record until (starting time is 0 s)
            nx = 800,         #number of positions we track
            nt = 85,          #number of time steps we take over the interval t_final - 0s
            savename='wave',   #name of file we save data to (will have an .npz appended to it)
    )

    animate_from_file("wave.npz", savename="../Animations/wave.gif", title="Wave Simulation", interval=100)
