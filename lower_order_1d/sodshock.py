from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

def sod_shock(cells, x_start, x_end, t_final, t_steps, savename):
    # Initialize independent and dependent variables
    x = np.linspace(x_start, x_end, 2 * cells - 1)
    dx = x[2] - x[0]

    rho = np.zeros_like(x)
    v = np.zeros_like(x)
    P = np.zeros_like(x)

    # Set up rho, v, and P in the container, excluding the "ghost cells" on the edges
    rho[x <= 0.5] = 10.
    rho[x > 0.5] = 1.
    P[x <= 0.5] = 8.
    P[x > 0.5] = 1.

    U, F, t, x = initialize(x, t_final, t_steps, rho, v, P)
    U = evolve(U, F, t, dx)
    if savename: save_data(savename, U, x, t)

sod_shock(cells = 400,
          x_start = 0.,
          x_end = 1.,
          t_final = 0.4,
          t_steps = 40,
          savename="testsodshock400")

# compare_files('sodshock200.npz', 'testsodshock200.npz')
# animate_from_file('testsodshock400.npz')
# animate_from_file('testsodshock800.npz')