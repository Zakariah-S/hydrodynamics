import numpy as np
import time

def apply_boundary_conditions(F, F_cells):
    #Reflective boundary conditions; the flux at the outer edges (outside the bounds of our x values)
    # is set to be the same as the innermost and outermost edge fluxes
    F[0] = F_cells[0]
    F[-1] = F_cells[-1]
    return F

def deconstruct_U(U, gamma=1.4):
    #Given the conserved value array U, get the corresponding densities, velocities, and pressures
    rho = U[:, 0]
    v = U[:, 1] / U[:, 0]
    p = (U[:, 2] - 0.5 * np.square(U[:, 1]) / U[:, 0]) * (gamma - 1.)
    return rho, v, p

def get_F_cells(rho, v, p, gamma=1.4):
    #Calculate the conserved-value-fluxes at cell centres, given rho, v, and p for those cells
    F = np.zeros((rho.size, 3))
    F[:, 0] = rho * v
    F[:, 1] = rho * np.square(v) + p
    F[:, 2] = v * (0.5 * rho * np.square(v) + p * gamma / (gamma - 1))
    return F

def get_alphas(rho, v, p, gamma=1.4):
    #Calculate alpha+ and alpha-, to be used to approximate edge fluxes and find the max time step length
    c_s = np.sqrt(gamma * p / rho)

    alpha_plus = np.zeros(rho.size - 1)
    alpha_minus = np.copy(alpha_plus)

    alpha_plus = np.max([np.zeros_like(alpha_plus, dtype=np.float64),  (v[:-1] + c_s[:-1]),  (v[1:] + c_s[1:])], axis=0)
    alpha_minus = np.max([np.zeros_like(alpha_minus, dtype=np.float64), -(v[:-1] - c_s[:-1]), -(v[1:] - c_s[1:])], axis=0)

    return alpha_plus, alpha_minus

def compute_time_step(U, dx, cfl=0.5, gamma=1.4):
    #Compute time step size dt based on CFL condition.
    rho, v, p = deconstruct_U(U)
    return cfl * dx / np.max(get_alphas(rho, v, p, gamma))

def step(U, F, dt, dx, gamma=1.4, cfl=0.5):
    if compute_time_step(U, dx, cfl, gamma) < dt:
        #If the default time step is too big, take two time steps that are half its size
        U, F = step(U, F, 0.5 * dt, dx, gamma=gamma, cfl=cfl)
        U, F = step(U, F, 0.5 * dt, dx, gamma=gamma, cfl=cfl)
        return U, F
    else: 
        L_U = -(F[1:, :] - F[:-1, :])/dx
        U_new = U + dt * L_U

        #Get rho, v, p at cell centres
        rho, v, p = deconstruct_U(U)

        #Get alpha lists
        alpha_plus, alpha_minus = get_alphas(rho, v, p)

        #Get fluxes at the centre of each cell
        F_cells = get_F_cells(rho, v, p)

        #Update fluxes at interfaces using HLL
        F[1:-1, :] = ((alpha_plus * F_cells[:-1, :].T + alpha_minus * F_cells[1:, :].T - 
                       alpha_plus * alpha_minus * (U_new[1:, :] - U_new[:-1, :]).T) / (alpha_plus + alpha_minus)).T
        F = apply_boundary_conditions(F, F_cells)

    return U_new, F

def evolve(U, F, t, dx, gamma=1.4, cfl=0.5):
    """
    Given the conserved values U[0] and F at the beginning of the simulation, return the array U
    where U[i] gives the conserved values at time t[i].

    U: numpy.ndarray[float], size = [#time steps + 1] x [#cells] x 3
        Array to record conserved values over the course of the simulation
    F: numpy.ndarray[float], size = [#cells + 1] x 3
        Array that holds fluxes of conserved values at the edge of each cell, also has "ghost values" 
        at the beginning and end that are used to enforce boundary conditions
    t: numpy.ndarray[float], size = [#time steps + 1]
        Times for which measurements will be recorded
    dx: float
        Distance between cell centres
    gamma: float
        Adiabatic constant, 1.4 for our purposes
    cfl: float
        Courant-Friedrich-Levy constant, 0.5 for our purposes
    """
    start_time = time.time()

    #Get time step length
    dt = t[1] - t[0]

    rho, v, p = deconstruct_U(U[0])
    F_cells = get_F_cells(rho, v, p, gamma)
    # Apply boundary conditions
    F = apply_boundary_conditions(F, F_cells)

    #Factor out dimensions
    #Get the normalization constants corresponding to each U_i
    n0 = np.mean(U[0, 2:-2, 0])
    n1 = n0 * dx / dt
    n2 = n0 * np.square(dx / dt)
    norms = np.array([n0, n1, n2])

    #Divide out the constants
    U /= norms

    #Divide out appropriate normalizations from F
    F_norms = np.array([n1, n2, n2 * dx/dt])
    F /= F_norms

    for i in range(1, t.size):
        U[i], F = step(U[i-1], F, dt=1., dx=1., gamma=gamma, cfl=cfl)
        print(f"t = {t[i]:.4f} s")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    return U * norms