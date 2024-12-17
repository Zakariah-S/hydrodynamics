from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

"""
Recommended: reflective boundary conditions. You can change this by editing the evolution.py file.
"""
def sod_shock(x_start, x_end, t_final, nx, nt, savename = None):
    # Initialize independent and dependent variables

    x = np.linspace(x_start, x_end, nx)
    v = np.zeros(nx)
    rho = np.ones_like(x)
    P = np.ones_like(x)

    rho[np.logical_and(x >= 0.45, x < 0.55)] = 100.
    P[np.logical_and(x >= 0.45, x < 0.55)] = 100.

    U, t, x = initialize(x, t_final, nt, rho, v, P)
    U = evolve(U, t, x[1] - x[0], nx)
    if savename: save_data(savename, U, x, t)

if __name__ == '__main__':
    sod_shock(x_start=0.,       #left side of tube
            x_end = 1.,       #right side of tube
            t_final = 0.40,    #time we record until (starting time is 0 s)
            nx = 800,         #number of positions we track
            nt = 40,          #number of time steps we take over the interval t_final - 0s
            savename='blast',   #name of file we save data to (will have an .npz appended to it)
    )

    animate_from_file("blast.npz", savename="../Animations/blast.gif", title="Blast Simulation", interval=100)