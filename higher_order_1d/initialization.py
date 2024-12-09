import numpy as np

def initialize(rho, v, P, t_final, gamma=1.4, theta=1.5):
    E = P / (gamma - 1) + 0.5 * rho * v ** 2
    rho_v = rho * v

    U = np.zeros((rho.size, 3))
    U[:, 0] = rho
    U[:, 1] = rho_v
    U[:, 2] = E

    # Time parameters
    t = 0.0
    t_final = 0.25
    cfl = 0.5

    return U