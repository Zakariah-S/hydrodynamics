from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

def wave(x_start, x_end, t_final, nx, nt, savename = None):
    # Initialize independent and dependent variables

    x = np.linspace(x_start, x_end, nx)
    rho = np.zeros(nx)
    v = np.zeros(nx)
    P = np.zeros(nx)

    rho = 10. + 0.00001 * np.sin(4 * np.pi * x)
    P = 10. + 0.00001 * np.sin(4 * np.pi * x)

    U, t, x = initialize(x, t_final, nt, rho, v, P)
    U = evolve(U, t, x[1] - x[0], nx)
    if savename: save_data(savename, U, x, t)

if __name__ == '__main__':
    wave(x_start=0.,       #left side of tube
            x_end = 1.,       #right side of tube
            t_final = 0.4,    #time we record until (starting time is 0 s)
            nx = 800,         #number of positions we track
            nt = 40,          #number of time steps we take over the interval t_final - 0s
            savename='testwave800')   #name of file we save data to (will have an .npz appended to it)
    
    animate_from_file("testwave800.npz", savename=None)