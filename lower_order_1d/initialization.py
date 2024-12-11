import numpy as np
from evolution import compute_time_step

def initialize(x, rho, v, p, t_final, steps, gamma=1.4, cfl=0.5):
    t = np.linspace(0., t_final, steps + 1)

    #Ensure that all arrays are oddly-sized so we can divide everything into cells
    if x.size % 2 == 0:
        for arr in [x, rho, v, p]:
            arr = arr[:-1]

    x = x[0::2]
    
    #Initialize U, using only the centres of each cell
    U = np.zeros((steps + 1, x.size, 3))
    U[0, :, 0] = rho[::2]
    U[0, :, 1] = (rho * v)[::2]
    U[0, :, 2] = (p / (gamma - 1) + 0.5 * rho * np.square(v))[::2]

    #Initialize F using only the edges of each cell
    F = np.zeros((x.size + 1, 3))
    F[1:-1, 0] = (rho * v)[1::2]
    F[1:-1, 1] = (rho * np.square(v) + p)[1::2]
    F[1:-1, 2] = (v * (0.5 * rho * np.square(v) + p * gamma / (gamma - 1)))[1::2]

    return U, F, t, x