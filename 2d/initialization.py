import numpy as np
from evolution import compute_time_step

def initialize(x, y, t_final, steps, rho, vx, vy, p, gamma=1.4, theta=1.5, cfl=0.5):
    t = np.linspace(0., t_final, steps + 1)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    #Initialize U
    U = np.zeros((steps + 1, 4, x.size, y.size))
    U[0, 0, :, :] = rho
    U[0, 1, :, :] = rho * vx
    U[0, 2, :, :] = rho * vy 
    U[0, 3, :, :] = p / (gamma - 1) + 0.5 * rho * (np.square(vx) + np.square(vy)) ** 2

    return U, t, x , y 