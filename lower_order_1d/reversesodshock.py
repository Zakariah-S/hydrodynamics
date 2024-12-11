from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

def reverse_sod_shock():
    # Initialize independent and dependent variables
    cells = 400
    x_start = 0.0
    x_end = 1.0
    x = np.linspace(x_start, x_end, 2 * cells - 1)
    dx = x[2] - x[0]

    rho = np.zeros_like(x)
    v = np.zeros_like(x)
    P = np.zeros_like(x)

    # Set up initial conditions for Sod shock tube
    rho_L = 10.0
    rho_R = 1.0
    P_L = 8.0
    P_R = 1.0
    v_L = 0.0
    v_R = 0.0

    # Set up rho, v, and P in the container, excluding the "ghost cells" on the edges
    rho[x <= 0.5] = rho_L
    rho[x > 0.5] = rho_R
    P[x <= 0.5] = P_L
    P[x > 0.5] = P_R
    v[:] = 0.

    #Simulation and plotting
    t_final = 0.40
    steps = 40

    U, F, t, x = initialize(x, rho[::-1], v[::-1], P[::-1], t_final, steps)
    U = evolve(U, F, t, dx)
    save_data("reversed_sodshock400", U, x, t)
# reverse_sod_shock()

# animate_from_file("reversed_sodshock400.npz")

t, x, rho, v, p = load_data("sodshock400.npz")
tr, xr, rhor, vr, pr = load_data("reversed_sodshock400.npz")

#If the arrays aren't mirror images of each other, find out for which indices this is true
# indices = np.argwhere(rho[1] != rhor[1, ::-1])
# print(indices)
# indices = np.argwhere(v[100] != -vr[100, ::-1])
# print(indices)
# indices = np.argwhere(p[1] != pr[1, ::-1])
# print(indices)

#Compare the original with the reversed
animate(t, x, rho - rhor[:, ::-1], np.abs(v) - np.abs(vr[:, ::-1]), p - pr[:, ::-1])