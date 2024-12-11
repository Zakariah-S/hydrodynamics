import numpy as np
import time

def apply_boundary_conditions(F):
    F[0] = F[1]
    F[-1] = F[-2]
    return F

def deconstruct_U(U, gamma=1.4):
    rho = U[:, 0]
    v = U[:, 1] / U[:, 0]
    p = (U[:, 2] - 0.5 * np.square(U[:, 1]) / U[:, 0]) * (gamma - 1.)
    return rho, v, p

def get_F_cells(rho, v, p, gamma=1.4):
    F = np.zeros((rho.size, 3))
    F[:, 0] = rho * v
    F[:, 1] = rho * np.square(v) + p
    F[:, 2] = v * (0.5 * rho * np.square(v) + p * gamma / (gamma - 1))
    return F

def get_alphas(rho, v, p, gamma=1.4):
    c_s = np.sqrt(gamma * p / rho)

    alpha_plus = np.zeros(rho.size - 1)
    alpha_minus = np.copy(alpha_plus)

    alpha_plus = np.max([np.zeros_like(alpha_plus, dtype=np.float64),  (v[:-1] + c_s[:-1]),  (v[1:] + c_s[1:])], axis=0)
    alpha_minus = np.max([np.zeros_like(alpha_minus, dtype=np.float64), -(v[:-1] - c_s[:-1]), -(v[1:] - c_s[1:])], axis=0)

    return alpha_plus, alpha_minus

def compute_time_step(U, dx, cfl=0.5, gamma=1.4):
    """
    Compute time step size dt based on CFL condition.
    """
    rho, v, p = deconstruct_U(U)
    return cfl * dx / np.max(get_alphas(rho, v, p, gamma))

def step(U, F, dt, dx, gamma=1.4, cfl=0.5):
    if compute_time_step(U, dx, cfl, gamma) < dt:
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
        F = apply_boundary_conditions(F)

    return U_new, F

def evolve(U, F, t, dx, gamma=1.4, cfl=0.5):
    start_time = time.time()

    # Apply boundary conditions
    F = apply_boundary_conditions(F)

    for i in range(1, t.size):
        U[i], F = step(U[i-1], F, t[i] - t[i-1], dx, gamma=gamma, cfl=cfl)
        print(f"t = {t[i]:.4f} s")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    return U