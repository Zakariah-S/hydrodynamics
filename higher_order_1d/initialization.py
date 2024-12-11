import numpy as np
from evolution import compute_time_step

def initialize(x, t_final, steps, rho, v, p, gamma=1.4, theta=1.5, cfl=0.5):
    t = np.linspace(0., t_final, steps + 1)
    dx = x[1] - x[0]

    #Expand domain to include ghost cells on the outside of the container
    x_expand = np.linspace(x[0] - 2*dx, x[-1] + 2*dx, x.size + 4)
    rho_expand = np.zeros(x_expand.size)
    v_expand = np.zeros(x_expand.size)
    p_expand = np.zeros(x_expand.size)
    rho_expand[2:-2] = rho
    v_expand[2:-2] = v
    p_expand[2:-2] = p

    #Initialize U
    U = np.zeros((steps + 1, x_expand.size, 3))
    U[0, :, 0] = rho_expand
    U[0, :, 1] = rho_expand * v_expand
    U[0, :, 2] = p_expand / (gamma - 1) + 0.5 * rho_expand * v_expand ** 2

    return U, t, x_expand