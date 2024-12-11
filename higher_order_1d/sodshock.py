from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

def sod_shock():
    # Initialize independent and dependent variables
    nx = 200

    x_start = 0.0
    x_end = 1.0
    dx = (x_end - x_start) / (nx - 1)
    x = np.linspace(x_start - 2 * dx, x_end + 2 * dx, nx + 4)  # Include ghost cells

    rho = np.zeros(nx + 4)
    v = np.zeros(nx + 4)
    P = np.zeros(nx + 4)

    # Set up initial conditions for Sod shock tube
    rho_L = 10.0
    rho_R = 1.0
    P_L = 8.0
    P_R = 1.0
    v_L = 0.0
    v_R = 0.0

    # Set up rho, v, and P in the container, excluding the "ghost cells" on the edges
    x_physical = x[2:-2]
    rho[2:-2][x_physical <= 0.5] = rho_L
    rho[2:-2][x_physical > 0.5] = rho_R
    P[2:-2][x_physical <= 0.5] = P_L
    P[2:-2][x_physical > 0.5] = P_R
    v[2:-2] = 0.

    #Simulation and plotting
    t_final = 0.40
    steps = 40

    U, t = initialize(rho, v, P, dx, t_final, steps)
    U = evolve(U, t, dx, nx)
    save_data("testsodshock", U, x, t)
# sod_shock()

animate_from_file("testsodshock.npz", interval=100)