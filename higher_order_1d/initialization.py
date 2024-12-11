import numpy as np
from evolution import compute_time_step

def initialize(rho, v, p, dx, t_final, steps, gamma=1.4, theta=1.5, cfl=0.5):
    t = np.linspace(0., t_final, steps + 1)
    
    #Initialize U
    U = np.zeros((steps + 1, rho.size, 3))
    U[0, :, 0] = rho
    U[0, :, 1] = rho * v
    U[0, :, 2] = p / (gamma - 1) + 0.5 * rho * v ** 2

    return U, t