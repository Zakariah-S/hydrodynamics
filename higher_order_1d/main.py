from initialization import initialize
from evolution import evolve
from output import plot_data
import numpy as np

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
v[2:-2][x_physical <= 0.5] = v_L
v[2:-2][x_physical > 0.5] = v_R

#Simulation and plotting
t_final = 0.10

U = initialize(rho, v, P, dx, t_final, gamma=1.4, theta=1.5)
U = evolve(U, t_final, dx, nx)
plot_data(U, x)