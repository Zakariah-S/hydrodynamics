import numpy as np
from evolution import compute_time_step

def initialize(rho, v, p, dx, t_final, gamma=1.4, theta=1.5, cfl=0.5):
    #Get size of time step
    c_s = np.sqrt(gamma * p / rho)[2:-2]
    max_speed = np.max(np.abs(v[2:-2]) + c_s)
    step_size = cfl * dx / max_speed

    steps = int(t_final / step_size) + 1

    #Initialize U
    U = np.zeros((steps + 1, rho.size, 3))
    U[0, :, 0] = rho
    U[0, :, 1] = rho * v
    U[0, :, 2] = p / (gamma - 1) + 0.5 * rho * v ** 2

    return U, step_size, steps