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

def reconstruct(c, theta):
    """
    Reconstruct left and right states at cell interfaces using minmod limiter.
    """
    nx, ny = c.shape

    c_Lx = np.zeros((nx + 1, ny))
    c_Rx = np.zeros((nx + 1, ny))
    c_Ly = np.zeros((nx, ny + 1))
    c_Ry = np.zeros((nx, ny + 1))
    sigma = np.zeros_like(c)

    #'left' and 'right' states along x-axis
    delta_c_minus =     c[2:nx-2,2:ny-2] - c[1:nx-3,2:ny-2]
    delta_c_plus =      c[3:nx-1,2:ny-2] - c[2:nx-2,2:ny-2]
    delta_c_center =    c[3:nx-1,2:ny-2] - c[1:nx-3,2:ny-2]
    sigma[2:nx-2,2:ny-2] = minmod(theta * delta_c_minus, 0.5 * delta_c_center, theta * delta_c_plus)

    c_Lx[2:nx-2, 2:ny-2] = c[2:nx-2, 2:ny-2] + 0.5 * sigma[2:ny-2, 2:ny-2]
    c_Rx[2:nx-2, 2:ny-2] = c[3:nx-1, 2:ny-2] - 0.5 * sigma[3:ny-1, 2:ny-2]

    #'left' and 'right' states along y-axis
    delta_c_minus =     c[2:nx-2,2:ny-2] - c[2:nx-2,1:ny-3]
    delta_c_plus =      c[2:nx-2,3:ny-1] - c[2:nx-2,2:ny-2]
    delta_c_center =    c[2:nx-2,3:ny-1] - c[2:nx-2,1:ny-3]
    sigma[2:nx-2,2:ny-2] = minmod(theta * delta_c_minus, 0.5 * delta_c_center, theta * delta_c_plus)

    c_Ly[2:nx-2, 2:ny-2] = c[2:nx-2, 2:ny-2] + 0.5 * sigma[2:nx-2, 2:ny-2]
    c_Ry[2:nx-2, 2:ny-2] = c[2:nx-2, 3:ny-1] - 0.5 * sigma[2:nx-2, 3:ny-1]

    return c_Lx, c_Rx, c_Ly, c_Ry

def construct_U(rho, vx, vy, p, gamma=1.4):
    U = np.zeros((rho.shape[0], rho.shape[1], 4))
    U[:, :, 0] = rho
    U[:, :, 1] = rho * vx
    U[:, :, 2] = rho * vy
    U[:, :, 3] = p/(gamma - 1.) + 0.5 * rho * (np.square(vx) + np.square(vy))
    return U

def deconstruct_U(U, gamma=1.4):
    """
    Get rho, v, and p from U
    """
    rho = U[:, :, 0]
    vx = U[:, :, 1] / U[:, :, 0]
    vy = U[:, :, 2] / U[:, :, 0]
    P = (gamma - 1) * (U[:, :, 3] - 0.5 * (np.square(U[:, :, 1]) + np.square(U[:, :, 2])) / U[:, :, 0])

    print(rho)
    print(vx)
    print(vy)

    if np.any(np.isnan(rho)):
        print('rho')
        exit()

    if np.any(np.isnan(vx)):
        print('vx')
        exit()

    if np.any(np.isnan(vy)):
        print('vy')
        exit()
    if np.any(np.isnan(P)):
        print('P')
        exit()
    return rho, vx, vy, P

def compute_F(rho, vx, vy, P, gamma=1.4):
    """
    Compute the flux vector F given conserved variables U.
    """
    shape = rho.shape
    F = np.zeros((shape[0], shape[1], 4))
    F[:, :, 0] = rho * vx
    F[:, :, 1] = rho * np.square(vx) + P
    F[:, :, 2] = rho * vx * vy
    F[:, :, 3] = (P*gamma/(gamma - 1.) + 0.5 * rho * (np.square(vx) + np.square(vy))) * vx

    return F

def compute_G(rho, vx, vy, P, gamma=1.4):
    shape = rho.shape
    G = np.zeros((shape[0], shape[1], shape[2], 4))
    G[:, :, 0] = rho * vy
    G[:, :, 1] = rho * vx * vy
    G[:, :, 2] = rho * np.square(vy) + P
    G[:, :, 3] = (P*gamma/(gamma - 1.) + 0.5 * rho * (np.square(vx) + np.square(vy))) * vy
    return G

def compute_hll_flux(U_L, U_R, axis, gamma=1.4):
    """
    Compute HLL flux at an interface given left and right states.
    """
    print("beginnign of fhll")
    rho_L, vx_L, vy_L, P_L = deconstruct_U(U_L)
    rho_R, vx_R, vy_R, P_R = deconstruct_U(U_R)

    # Apply pressure and density floors to prevent negative values
    P_L = np.maximum(P_L, 1e-6)
    P_R = np.maximum(P_R, 1e-6)
    rho_L = np.maximum(rho_L, 1e-6)
    rho_R = np.maximum(rho_R, 1e-6)

    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)

    if axis == 'x':
        S_L = np.min([vx_L - c_L, vx_R - c_R, np.zeros_like(vx_L)], axis=0)
        S_R = np.max([vx_L + c_L, vx_R + c_R, np.zeros_like(vx_R)], axis=0)

        F_HLL = np.zeros_like(U_L)

        F_L = compute_F(rho_L, vx_L, P_R, gamma)
        F_R = compute_F(rho_L, vx_L, P_L, gamma)

    elif axis == 'y':
        S_L = np.min([vy_L - c_L, vy_R - c_R, np.zeros_like(vy_L)], axis=0)
        S_R = np.max([vy_L + c_L, vy_R + c_R, np.zeros_like(vy_R)], axis=0)

        F_HLL = np.zeros_like(U_L)

        F_L = compute_F(rho_L, vy_L, P_R, gamma)
        F_R = compute_F(rho_L, vy_L, P_L, gamma)

    for i in np.arange(3):
        F_HLL[:, :, i] = ((F_L[:, :, i] * S_R - F_R[:, :, i] * S_L + (U_R - U_L)[:, :, i] * S_L * S_R) / (S_R - S_L))

    if np.any(np.isnan(F_HLL)):
        print('end of hll')
        exit()

    return F_HLL

def compute_L(U, nx, ny, dx, dy, gamma=1.4, theta=1.5):
    """
    Compute the spatial derivative operator L(U).
    """
    L = np.zeros_like(U)

    print("beginning of L")
    rho, vx, vy, P = deconstruct_U(U)

    # Reconstruct variables
    rho_Lx, rho_Rx, rho_Ly, rho_Ry = reconstruct(rho, theta)
    vx_Lx, vx_Rx, vx_Ly, vx_Ry = reconstruct(vx, theta)
    vy_Lx, vy_Rx, vy_Ly, vy_Ry = reconstruct(vy, theta)
    P_Lx, P_Rx, P_Ly, P_Ry = reconstruct(P, theta)

    #Get F
    Ux_Lx = np.zeros((nx + 1, ny, 4))
    Ux_Rx = np.zeros((nx + 1, ny, 4))

    Ux_Lx[2:nx-2, :, :] = construct_U(rho_Lx[2:nx-2, :], vx_Lx[2:nx-2, :], vy_Lx[2:nx-2, :], P_Lx[2:nx-2, :])
    Ux_Rx[2:nx-2, :, :] = construct_U(rho_Rx[2:nx-2, :], vx_Rx[2:nx-2, :], vy_Lx[2:nx-2, :], P_Rx[2:nx-2, :])

    F = np.zeros((nx + 1, ny, 4))
    F[2:nx-2, :, :] = compute_hll_flux(Ux_Lx[2:nx-2, :, :], Ux_Rx[2:nx-2, :, :], 'x', gamma)
    
    if np.any(np.isnan(F)):
        print('end of L compute')
        exit()

    #Get G
    Ux_Ly = np.zeros((nx, ny + 1, 4))
    Ux_Ry = np.zeros((nx, ny + 1, 4))

    Ux_Ly[:, 2:ny-2, :] = construct_U(rho_Ly[:, 2:ny-2], vx_Ly[:, 2:ny-2], vy_Ly[:, 2:ny-2], P_Ly[:, 2:ny-2])
    Ux_Ry[:, 2:ny-2, :] = construct_U(rho_Ry[:, 2:ny-2], vx_Ry[:, 2:ny-2], vy_Ly[:, 2:ny-2], P_Ry[:, 2:ny-2])

    G = np.zeros((nx, ny + 1, 4))
    G[:, 2:ny-2, :] = compute_hll_flux(Ux_Ly[:, 2:ny-2, :], Ux_Ry[:, 2:ny-2, :], 'y', gamma)

    if np.any(np.isnan(G)):
        print('end of G')
        exit()

    L[3:nx-3, 3:ny-3:, :] = - (F[3:nx-3, 3:ny-3, :] - F[2:nx-4, 3:ny-3, :]) / dx - (G[3:nx-3, 3:ny-3, :] - G[3:nx-3, 2:ny-4, :]) / dy
 
    if np.any(np.isnan(L)):
        print('end of L compute')
        exit()
    return L

def apply_boundary_conditions(U):
    """
    Apply periodic boundary conditions in both directions.
    """
    U[0, :, :] = U[3, :, :]
    U[1, :, :] = U[2, :, :]
    U[-1, :, :] = U[-4, :, :]
    U[-2, :, :] = U[-3, :, :]

    U[:, 0, :] = U[:, 3, :]
    U[:, 1, :] = U[:, 2, :]
    U[:, -1, :] = U[:, -4, :]
    U[:, -2, :] = U[:, -3, :]

    return U

def shu_osher(U, dt, dx, dy, nx, ny, gamma=1.4, theta=1.5):
    """
    Perform a time step using the Shu-Osher third-order scheme.
    """
    # Stage 1
    L1 = compute_L(U, nx, ny, dx, dy, gamma, theta)
    U1 = U + dt * L1
    U1 = apply_boundary_conditions(U1)

    if np.any(np.isnan(U1)):
        print('shu osher 1')
        exit()

    # Stage 2
    L2 = compute_L(U1, nx, ny, dx, dy, gamma, theta)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    U2 = apply_boundary_conditions(U2)

    if np.any(np.isnan(U1)):
        print('shu osher 2')
        exit()

    # Stage 3
    L3 = compute_L(U2, nx, ny, dx, dy, gamma, theta)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)
    U_new = apply_boundary_conditions(U_new)

    if np.any(np.isnan(U_new)):
        print('shu osher 3')
        exit()

    return U_new

def compute_time_step(U, dx, cfl, gamma):
    """
    Compute time step size dt based on CFL condition.
    """
    print("time")
    rho, vx, vy, P = deconstruct_U(U)

    # Apply pressure and density floors
    P = np.maximum(P, 1e-6)
    rho = np.maximum(rho, 1e-6)

    c_s = np.sqrt(gamma * P / rho)
    max_speed = np.max(np.hypot(vx, vy) + c_s)
    dt = cfl * dx / max_speed
    return dt

def step(U, dt, dx, dy, nx, ny, gamma=1.4, cfl=0.5, theta=1.5):
    if compute_time_step(U, dx, cfl, gamma) < dt:
        U = step(U, 0.5 * dt, dx, dy, nx, ny, gamma=1.4, cfl=0.5, theta=1.5)
        U = step(U, 0.5 * dt, dx, dy, nx, ny, gamma=1.4, cfl=0.5, theta=1.5)
        return U
    else: 
        U = shu_osher(U, dt, dx, dy, nx, ny, gamma, theta)
    return U

def evolve(U, t, dx, dy, nx, ny, gamma=1.4, cfl=0.5, theta=1.5):
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
    assert dx == dy
    # Apply boundary conditions
    U[0] = apply_boundary_conditions(U[0])

    #Factor out dimensions
    #Get the normalization constants corresponding to each U_i
    n0 = np.mean(U[0, 2:-2, 2:-2, 0])
    n1 = n0 * dx / dt
    n2 = n1
    n3 = n0 * np.square(dx / dt)
    norms = np.array([n0, n1, n2, n3])

    #Divide out the constants
    U = U / norms

    for i in range(1, t.size):
        U[i] = step(U[i-1], dt=1., dx=1., dy=1., nx=nx + 4, ny=ny + 4) #Divide out dt and dx from time step & cell length to make them dimensionless
        print(t[i])

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    return U * norms #put the dimensions back