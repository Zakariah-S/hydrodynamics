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

    rho[x <= 0.5] = 10. #left side density
    rho[x > 0.5] = 1. #right side density
    P[x <= 0.5] = 8. #left side density
    P[x > 0.5] = 1. #right side density

    U, t, x = initialize(x, t_final, nt, rho, v, P)
    U = evolve(U, t, x[1] - x[0], nx)
    if savename: save_data(savename, U, x, t)

if __name__ == '__main__':
    sod_shock(x_start=0.,       #left side of tube
            x_end = 1.,       #right side of tube
            t_final = 0.4,    #time we record until (starting time is 0 s)
            nx = 800,         #number of positions we track
            nt = 40,          #number of time steps we take over the interval t_final - 0s
            savename='testsodshock800')   #name of file we save data to (will have an .npz appended to it)
    
    # residuals_animation('testsodshock200.npz', '../exactsodshock1d/exactsodshock200.npz')

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
