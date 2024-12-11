import numpy as np
from evolution import compute_time_step

def initialize(x, t_final, steps, rho, v, p, gamma=1.4, cfl=0.5):
    """
    - Generate the set of times t at which we will make measurements
    - Initialize an array U that will record the system's conserved values at each cell centre, for all times in t
    - Initialize an array F that holds the initial fluxes at the edge of each cell
    - Return an array x which holds only the x values of each cell centre

    x: numpy.ndarray[float]
        Should have a size equal to 2 * #cells + 1, such that x also contains the positions of each cell's edges
    rho, v, p: numpy.ndarray[float]
        Initial values of density, velocity, and pressure at each point x (in the original x array)
    t_final: float
        Run time of simulation
    steps: int
        Number of time steps that will be taken to get from 0 to t_final
    gamma: float
        Adiabatic constant, 1.4 for our purposes
    cfl: float
        Courant-Friedrich-Levy constant, 0.5 for our purposes
    """
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