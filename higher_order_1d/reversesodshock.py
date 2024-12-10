from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

def reverse_sod_shock():
    # Initialize independent and dependent variables
    nx = 400

    x_start = 0.0
    x_end = 1.0
    dx = (x_end - x_start) / (nx - 1)
    x = np.linspace(x_start - 2 * dx, x_end + 2 * dx, nx + 4)  # Include ghost cells

    rho = np.zeros(nx + 4)
    v = np.zeros(nx + 4)
    P = np.zeros(nx + 4)

    # Set up initial conditions for Sod shock tube
    rho_R = 10.0
    rho_L = 1.0
    P_R = 8.0
    P_L = 1.0
    v_R = 0.0
    v_L = 0.0

    # Set up rho, v, and P in the container, excluding the "ghost cells" on the edges
    x_physical = x[2:-2]
    rho[2:-2][x_physical <= 0.5] = rho_L
    rho[2:-2][x_physical > 0.5] = rho_R
    P[2:-2][x_physical <= 0.5] = P_L
    P[2:-2][x_physical > 0.5] = P_R
    v[2:-2] = 0.

    #Simulation and plotting
    t_final = 0.25

    U, step_size, steps = initialize(rho, v, P, dx, t_final)
    U = evolve(U, step_size, t_final, dx, nx)
    t = step_size * np.arange(steps + 1)
    # # plot_from_one_U(U[-1], x)
    save_data("reversed_sodshock", U, x, t)
reverse_sod_shock()

t, x, rho, v, p = load_data("sodshock.npz")
tr, xr, rhor, vr, pr = load_data("reversed_sodshock.npz")
# plot_one_time(x, rho[-1], v[-1], p[-1], t[-1])

#Compare the original with the reversed
animate(t, x, rho - rhor[:, ::-1], np.abs(v) - np.abs(vr[:, ::-1]), p - pr[:, ::-1])