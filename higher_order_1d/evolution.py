import numpy as np
import time

def minmod(a, b, c):
    """
    Minmod function for three arguments as per equation (12).
    """
    sgn_a = np.sign(a)
    sgn_b = np.sign(b)
    sgn_c = np.sign(c)
    min_abs = np.min([np.abs(a), np.abs(b), np.abs(c)], axis=0)
    result = 0.25 * np.abs(sgn_a + sgn_b) * (sgn_a + sgn_c) * min_abs
    return result

def reconstruct(c, theta):
    """
    Reconstruct left and right states at cell interfaces using minmod limiter.
    """
    # nx = len(c)
    # c_L = np.zeros(nx + 1)  # Interfaces from 0 to nx
    # c_R = np.zeros(nx + 1)
    # sigma = np.zeros(nx)

    # delta_c_minus = c[2:nx-2] - c[1:nx-3]
    # delta_c_plus = c[3:nx-1] - c[2:nx-2]
    # delta_c_center = 0.5 * (c[3:nx-1] - c[1:nx-3])
    # sigma[2:nx-2] = minmod(theta * delta_c_minus, delta_c_center, theta * delta_c_plus)

    # c_L[2:nx-2] = c[2:nx-2] + 0.5 * sigma[2:nx-2]
    # c_R[2:nx-2] = c[3:nx-1] - 0.5 * sigma[3:nx-1]

    c_L = np.zeros(c.size + 1)
    c_R = np.zeros(c.size + 1)
    sigma = np.zeros_like(c)

    delta_c_minus = c[2:-3] - c[1:-4]
    delta_c_plus = c[3:-2] - c[2:-3]
    delta_c_center = 0.5 * (c[3:-2] - c[1:-4])
    sigma[2:-3] = minmod(theta * delta_c_minus, delta_c_center, theta * delta_c_plus)

    c_L[2:-4] = c[2:-3] + 0.5 * sigma[2:-3]
    c_R[2:-4] = c[3:-2] - 0.5 * sigma[3:-2]

    return c_L, c_R

def construct_U(rho, v, P, gamma=1.4):
    U = np.zeros((rho.size, 3))
    U[:, 0] = rho

def deconstruct_U(U, gamma=1.4):
    """
    Get rho, v, and p from U
    """
    rho = U[:, 0]
    v = U[:, 1] / U[:, 0]
    P = (gamma - 1) * (U[:, 2] - 0.5 * np.square(U[:, 1]) / U[:, 0])
    return rho, v, P

def compute_flux(U, gamma):
    """
    Compute the flux vector F given conserved variables U.
    """
    rho, v, P = deconstruct_U(U, gamma)
    E = U[:, 2]
    F = np.zeros_like(U)
    F[:, 0] = rho * v
    F[:, 1] = rho * v ** 2 + P
    F[:, 2] = (E + P) * v
    return F

def compute_hll_flux(U_L, U_R, gamma):
    """
    Compute HLL flux at an interface given left and right states.
    """
    rho_L, v_L, P_L = deconstruct_U(U_L)
    rho_R, v_R, P_R = deconstruct_U(U_R)

    # Apply pressure and density floors to prevent negative values
    P_L = np.maximum(P_L, 1e-6)
    P_R = np.maximum(P_R, 1e-6)
    rho_L = np.maximum(rho_L, 1e-6)
    rho_R = np.maximum(rho_R, 1e-6)

    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)

    S_L = np.min([v_L - c_L, v_R - c_R, np.zeros_like(v_L)], axis=0)
    S_R = np.max([v_L + c_L, v_R + c_R, np.zeros_like(v_R)], axis=0)

    F_HLL = np.zeros_like(U_L)

    F_L = compute_flux(U_L, gamma).T
    F_R = compute_flux(U_R, gamma).T

    F_HLL = ((S_R * F_L - S_L * F_R + S_L * S_R * (U_R.T - U_L.T)) / (S_R - S_L)).T

    return F_HLL

def compute_L(U, nx, dx, gamma, theta):
    """
    Compute the spatial derivative operator L(U).
    """
    L = np.zeros_like(U)
    rho = U[:, 0]
    rho_v = U[:, 1]
    E = U[:, 2]
    v = rho_v / rho
    P = (gamma - 1) * (E - 0.5 * rho * v ** 2)

    # Reconstruct variables
    rho_L, rho_R = reconstruct(rho, theta)
    v_L, v_R = reconstruct(v, theta)
    P_L, P_R = reconstruct(P, theta)

    U_L = np.zeros((nx + 1, 3))
    U_R = np.zeros((nx + 1, 3))
    F = np.zeros((nx + 1, 3))

    U_L[2:nx-2, 0] = rho_L[2:nx-2]
    U_L[2:nx-2, 1] = rho_L[2:nx-2] * v_L[2:nx-2]
    U_L[2:nx-2, 2] = P_L[2:nx-2] / (gamma - 1) + 0.5 * rho_L[2:nx-2] * v_L[2:nx-2] ** 2
    U_R[2:nx-2, 0] = rho_R[2:nx-2]
    U_R[2:nx-2, 1] = rho_R[2:nx-2] * v_R[2:nx-2]
    U_R[2:nx-2, 2] = P_R[2:nx-2] / (gamma - 1) + 0.5 * rho_R[2:nx-2] * v_R[2:nx-2] ** 2

    F[2:nx-2] = compute_hll_flux(U_L[2:nx-2], U_R[2:nx-2], gamma)
    L[3:nx-3] = - (F[3:nx-3] - F[2:nx-4]) / dx

    return L

def apply_boundary_conditions(U):
    """
    Apply boundary conditions to U.
    """
    # Reflective boundary conditions
    U[0, :] = U[3, :]
    U[1, :] = U[2, :]
    U[-1, :] = U[-4, :]
    U[-2, :] = U[-3, :]
    return U

def shu_osher(U, nx, dx, dt, gamma, theta):
    """
    Perform a time step using the Shu-Osher third-order scheme.
    """
    # Stage 1
    L1 = compute_L(U, nx, dx, gamma, theta)
    U1 = U + dt * L1
    U1 = apply_boundary_conditions(U1)

    # Stage 2
    L2 = compute_L(U1, nx, dx, gamma, theta)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    U2 = apply_boundary_conditions(U2)

    # Stage 3
    L3 = compute_L(U2, nx, dx, gamma, theta)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)
    U_new = apply_boundary_conditions(U_new)

    return U_new

def compute_time_step(U, dx, cfl, gamma):
    """
    Compute time step size dt based on CFL condition.
    """
    rho = U[:, 0]
    rho_v = U[:, 1]
    E = U[:, 2]
    v = rho_v / rho
    P = (gamma - 1) * (E - 0.5 * rho * v ** 2)

    # Apply pressure and density floors
    P = np.maximum(P, 1e-6)
    rho = np.maximum(rho, 1e-6)

    c_s = np.sqrt(gamma * P / rho)
    max_speed = np.max(np.abs(v) + c_s)
    dt = cfl * dx / max_speed
    return dt

def step(U, dt, dx, nx, gamma=1.4, cfl=0.5, theta=1.5):
    if compute_time_step(U, dx, cfl, gamma) < dt:
        U = step(U, 0.5 * dt, dx, nx, gamma=1.4, cfl=0.5, theta=1.5)
        U = step(U, 0.5 * dt, dx, nx, gamma=1.4, cfl=0.5, theta=1.5)
        return U
    else: 
        U = shu_osher(U, nx, dx, dt, gamma, theta)
    return U

def evolve(U, t, dx, nx, gamma=1.4, cfl=0.5, theta=1.5):
    start_time = time.time()

    # Apply boundary conditions
    U[0] = apply_boundary_conditions(U[0])
    t_final = t[-1]

    for i in range(1, t.size):
        U[i] = step(U[i-1], t[i] - t[i-1], dx, nx + 4)
        print(f"t = {t[i]:.4f} s")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    return U