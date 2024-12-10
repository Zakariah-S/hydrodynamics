import numpy as np
from evolution import compute_time_step

def initialize(rho, v, p, dx, t_final, gamma=1.4, theta=1.5, cfl=0.5):
    #Get number of steps we want to take
    c_s = np.sqrt(gamma * p / rho)
    max_speed = np.max(np.abs(v) + c_s)
    dt = cfl * dx / max_speed

    #Initialize U
    U = np.zeros((rho.size, 3))
    U[:, 0] = rho
    U[:, 1] = rho * v
    U[:, 2] = p / (gamma - 1) + 0.5 * rho * v ** 2

    return U