from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

def sod_shock(x_start, x_end, y_start, y_end, t_final, nx, ny, nt, savename = None):
    # Initialize independent and dependent variables

    x = np.linspace(x_start, x_end, nx)
    y = np.linspace(y_start, y_end, ny)
    rho = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    P = np.zeros((nx, ny))

    rho[x <= 0.5, :] = 10. #left side density
    rho[x > 0.5, :] = 1. #right side density
    P[x <= 0.5, :] = 8. #left side density
    P[x > 0.5, :] = 1. #right side density

    U, t, x, y = initialize(x, y, t_final, nt, rho, v, P)
    U = evolve(U, t, x[1] - x[0], y[1] - y[0], nx, ny)
    if savename:
        save_data(savename, U, x, y, t)

if __name__ == '__main__':
    # sod_shock(x_start=0.,
    #         x_end = 1.,
    #         y_start=0.,
    #         y_end = 1.,
    #         t_final = 0.4,    #time we record until (starting time is 0 s)
    #         nx = 50,         #number of positions along x that we track
    #         ny = 50,         #number of positions along y that we track
    #         nt = 40,          #number of time steps we take over the interval t_final - 0s
    #         savename='testsodshock50x50')   #name of file we save data to (will have an .npz appended to it)
    pass

    # animate_from_file("testsodshock10x10.npz")
    