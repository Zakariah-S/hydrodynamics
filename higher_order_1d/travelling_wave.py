from initialization import initialize
from evolution import evolve
from output import *
import numpy as np
import sys

def sod_shock(x_start, x_end, t_final, nx, nt, savename = None):
    # Initialize independent and dependent variables

    x = np.linspace(x_start, x_end, nx)
    rho = np.zeros(nx)
    v = np.zeros(nx)
    P = np.zeros(nx)

    # rho[:] = 1.
    # rho[3 * nx // 8:5*nx //8] = 100.
    # P[:] = 1.
    # P[3 * nx // 8:5*nx //8] = 100.
    
    rho = 3. + np.sin(8 * np.pi * x)
    P = 4. + np.sin(8 * np.pi * x)

    U, t, x = initialize(x, t_final, nt, rho, v, P)
    U = evolve(U, t, x[1] - x[0], nx)
    if savename: save_data(savename, U, x, t)

if __name__ == '__main__':
    sod_shock(x_start=0.,       #left side of tube
            x_end = 1.,       #right side of tube
            t_final = 1.,    #time we record until (starting time is 0 s)
            nx = 800,         #number of positions we track
            nt = 100,          #number of time steps we take over the interval t_final - 0s
            savename='wave',   #name of file we save data to (will have an .npz appended to it)
    )

    animate_from_file("wave.npz", interval=100)

    # animate_from_file("testsodshock200.npz")
    # sod_shock(x_start=0.,       #left side of tube
    #           x_end = 1.,       #right side of tube
    #           t_final = 0.4,    #time we record until (starting time is 0 s)
    #           nx = 400,         #number of positions we track
    #           nt = 40,          #number of time steps we take over the interval t_final - 0s
    #           savename='testsodshock400')   #name of file we save data to (will have an .npz appended to it)

    # sod_shock(x_start=0.,       #left side of tube
    #           x_end = 1.,       #right side of tube
    #           t_final = 0.4,    #time we record until (starting time is 0 s)
    #           nx = 800,         #number of positions we track
    #           nt = 40,          #number of time steps we take over the interval t_final - 0s
    #           savename='testsodshock800')   #name of file we save data to (will have an .npz appended to it)

    # # Default parameters
    # default_cells = 400
    # cells = int(sys.argv[1]) if len(sys.argv) > 1 else default_cells


    # sod_shock(x_start = 0.,
    #           x_end = 1.,
    #           t_final = 0.4,
    #           nx = cells,
    #           nt = 40,
    #           savename=f"testsodshock{cells}")
