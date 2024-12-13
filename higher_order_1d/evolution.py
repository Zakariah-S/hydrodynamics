"""
Module with functions that guide the evolution of the 1-D fluid system, with lower-order errors.
"""
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

def get_left_and_right_states(c, theta):
    """
    Reconstruct left and right states at cell interfaces using minmod limiter.
    """
    c_L = np.zeros(c.size + 1)
    c_R = np.zeros(c.size + 1)
    sigma = np.zeros_like(c)

    #sigma is 0 at the edge cells because of no-slip boundary conditions
    sigma[1:-1] = 0.5 * minmod(theta * (c[1:-1] - c[:-2]),
                               0.5 * (c[2:] - c[:-2]),
                               theta * (c[2:] - c[1:-1]))
    
    c_L[1:] = c + 0.5 * sigma
    c_L[0] = c[0]

    c_R[:-1] = c - 0.5 * sigma
    c_R[-1] = c[-1]

    return c_L, c_R

def construct_U(rho, v, p, gamma=1.4):
    U = np.array([
        rho,
        rho * v,
        p/(gamma - 1.) + 0.5 * rho * np.square(v)
    ])
    return U

def deconstruct_U(U, gamma=1.4):
    """
    Get rho, v, and p from U
    """
    rho = U[0]
    v = U[1] / U[0]
    P = (gamma - 1) * (U[2] - 0.5 * np.square(U[1]) / U[0])
    return rho, v, P

def compute_flux(rho, v, P, gamma):
    """
    Compute the flux vector F given conserved variables U.
    """
    F = np.array([
        rho * v, 
        rho * np.square(v) + P,
        (P * (gamma/(gamma - 1.)) + 0.5 * rho * np.square(v)) * v
    ])
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

    F_L = compute_flux(rho_L, v_L, P_L, gamma)
    F_R = compute_flux(rho_R, v_R, P_R, gamma)

    F_HLL = ((S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L))

    return F_HLL

def compute_L(U, nx, dx, gamma, theta):
    """
    Compute the spatial derivative operator L(U).
    """
    L = np.zeros_like(U)

    rho, v, P = deconstruct_U(U)

    # Reconstruct variables
    rho_L, rho_R = get_left_and_right_states(rho, theta)
    v_L, v_R = get_left_and_right_states(v, theta)
    P_L, P_R = get_left_and_right_states(P, theta)

    U_L = construct_U(rho_L, v_L, P_L)
    U_R = construct_U(rho_R, v_R, P_R)

    F = compute_hll_flux(U_L, U_R, gamma)

    L = - (F[:, 1:] - F[:, :-1]) / dx

    return L

def shu_osher(U, nx, dx, dt, gamma, theta):
    """
    Perform a time step using the Shu-Osher third-order scheme.
    """
    # Stage 1
    L1 = compute_L(U, nx, dx, gamma, theta)
    U1 = U + dt * L1

    # Stage 2
    L2 = compute_L(U1, nx, dx, gamma, theta)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)

    # Stage 3
    L3 = compute_L(U2, nx, dx, gamma, theta)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)

    return U_new

def compute_time_step(U, dx, cfl, gamma):
    """
    Compute time step size dt based on CFL condition.
    """
    rho, v, P = deconstruct_U(U)

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
    """
    Given the array of conserved values U[0] at the beginning of the simulation, return the array U
    where U[i] gives the conserved values for all x at time t[i].

    U: numpy.ndarray[float], size = [#time steps + 1] x [#cells] x 3
        Array to record conserved values over the course of the simulation
    t: numpy.ndarray[float], size = [#time steps + 1]
        Times for which measurements will be recorded
    dx: float
        Distance between cell centres
    nx: Number of recorded positions x
    gamma: float
        Adiabatic constant, 1.4 for our purposes
    cfl: float
        Courant-Friedrich-Levy constant, 0.5 for our purposes
    theta: float
        Constant associated with the minmod function, must be between 1 and 2 (we chose 1.5)
    """
    start_time = time.time()

    dt = t[1] - t[0]

    #Factor out dimensions
    #Get the normalization constants corresponding to each U_i
    n0 = np.mean(U[0, 0])
    n1 = n0 * dx / dt
    n2 = n0 * np.square(dx / dt)
    norms = np.array([n0, n1, n2])

    #Divide out the constants
    U[0] = (U[0].T / norms).T

    for i in range(1, t.size):
        U[i] = step(U[i-1], dt=1., dx=1., nx=nx) #Divide out dt and dx from time step & cell length to make them dimensionless

        # print(f"t = {t[i]:.4f} s")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    # print((norms[:,None] * U).shape)
    return norms[:,None] * U #put the dimensions back