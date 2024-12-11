import numpy as np
import matplotlib.pyplot as plt
import time

def minmod(a, b, c):
    """
    Minmod function for three arguments as per equation (12).
    """
    sgn_a = np.sign(a)
    sgn_b = np.sign(b)
    sgn_c = np.sign(c)
    min_abs = np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
    result = 0.25 * np.abs(sgn_a + sgn_b) * (sgn_a + sgn_c) * min_abs
    return result

def reconstruct_2d(c, theta, axis):
    """
    Reconstruct left and right states at cell interfaces using minmod limiter.
    """
    c_L = np.zeros_like(c)
    c_R = np.zeros_like(c)
    sigma = np.zeros_like(c)

    if axis == 0:
        # Reconstruction in x-direction
        for i in range(2, c.shape[0] - 2):
            delta_c_minus = c[i, :] - c[i - 1, :]
            delta_c_plus = c[i + 1, :] - c[i, :]
            delta_c_center = 0.5 * (c[i + 1, :] - c[i - 1, :])
            sigma[i, :] = minmod(theta * delta_c_minus, delta_c_center, theta * delta_c_plus)
        for i in range(2, c.shape[0] - 2):
            c_L[i, :] = c[i, :] + 0.5 * sigma[i, :]
            c_R[i, :] = c[i + 1, :] - 0.5 * sigma[i + 1, :]
    elif axis == 1:
        # Reconstruction in y-direction
        for j in range(2, c.shape[1] - 2):
            delta_c_minus = c[:, j] - c[:, j - 1]
            delta_c_plus = c[:, j + 1] - c[:, j]
            delta_c_center = 0.5 * (c[:, j + 1] - c[:, j - 1])
            sigma[:, j] = minmod(theta * delta_c_minus, delta_c_center, theta * delta_c_plus)
        for j in range(2, c.shape[1] - 2):
            c_L[:, j] = c[:, j] + 0.5 * sigma[:, j]
            c_R[:, j] = c[:, j + 1] - 0.5 * sigma[:, j + 1]
    else:
        raise ValueError("Axis must be 0 (x) or 1 (y)")

    return c_L, c_R

def compute_flux(U, gamma, axis):
    """
    Compute the flux vector F or G given conserved variables U.
    """
    rho = U[:, :, 0]
    rho_vx = U[:, :, 1]
    rho_vy = U[:, :, 2]
    E = U[:, :, 3]

    print('.')
    v_x = rho_vx / rho
    v_y = rho_vy / rho
    print('.')
    P = (gamma - 1) * (E - 0.5 * rho * (v_x ** 2 + v_y ** 2))

    F = np.zeros_like(U)
    if axis == 0:
        # Flux in x-direction
        F[:, :, 0] = rho_vx
        F[:, :, 1] = rho_vx * v_x + P
        F[:, :, 2] = rho_vx * v_y
        F[:, :, 3] = (E + P) * v_x
    elif axis == 1:
        # Flux in y-direction
        F[:, :, 0] = rho_vy
        F[:, :, 1] = rho_vy * v_x
        F[:, :, 2] = rho_vy * v_y + P
        F[:, :, 3] = (E + P) * v_y
    else:
        raise ValueError("Axis must be 0 (x) or 1 (y)")

    return F

def compute_hll_flux(U_L, U_R, gamma, axis):
    """
    Compute HLL flux at interfaces given left and right states.
    """
    rho_L = U_L[:, :, 0]
    rho_R = U_R[:, :, 0]
    rho_vx_L = U_L[:, :, 1]
    rho_vx_R = U_R[:, :, 1]
    rho_vy_L = U_L[:, :, 2]
    rho_vy_R = U_R[:, :, 2]
    E_L = U_L[:, :, 3]
    E_R = U_R[:, :, 3]

    # Prevent division by zero
    rho_L = np.maximum(rho_L, 1e-8)
    rho_R = np.maximum(rho_R, 1e-8)

    v_x_L = rho_vx_L / rho_L
    v_x_R = rho_vx_R / rho_R
    v_y_L = rho_vy_L / rho_L
    v_y_R = rho_vy_R / rho_R

    P_L = (gamma - 1) * (E_L - 0.5 * rho_L * (v_x_L ** 2 + v_y_L ** 2))
    P_R = (gamma - 1) * (E_R - 0.5 * rho_R * (v_x_R ** 2 + v_y_R ** 2))

    # Apply pressure floors
    P_L = np.maximum(P_L, 1e-8)
    P_R = np.maximum(P_R, 1e-8)

    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)

    if axis == 0:
        v_L = v_x_L
        v_R = v_x_R
    elif axis == 1:
        v_L = v_y_L
        v_R = v_y_R
    else:
        raise ValueError("Axis must be 0 (x) or 1 (y)")

    lambda_L_minus = v_L - c_L
    lambda_L_plus = v_L + c_L
    lambda_R_minus = v_R - c_R
    lambda_R_plus = v_R + c_R

    alpha_plus = np.maximum.reduce([np.zeros_like(lambda_L_plus), lambda_L_plus, lambda_R_plus])
    alpha_minus = np.maximum.reduce([np.zeros_like(lambda_L_minus), -lambda_L_minus, -lambda_R_minus])

    denominator = alpha_plus + alpha_minus
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)

    F_L = compute_flux(U_L, gamma, axis)
    F_R = compute_flux(U_R, gamma, axis)

    F_HLL = (alpha_plus[..., np.newaxis] * F_L + alpha_minus[..., np.newaxis] * F_R - alpha_plus[..., np.newaxis] * alpha_minus[..., np.newaxis] * (U_R - U_L)) / denominator[..., np.newaxis]

    return F_HLL

def compute_L(U, dx, dy, gamma, theta):
    """
    Compute the spatial derivative operator L(U) in 2D.
    """
    L = np.zeros_like(U)

    # Extract conserved variables
    rho = U[:, :, 0]
    rho_vx = U[:, :, 1]
    rho_vy = U[:, :, 2]
    E = U[:, :, 3]

    # Prevent division by zero
    rho = np.maximum(rho, 1e-8)

    v_x = rho_vx / rho
    v_y = rho_vy / rho

    P = (gamma - 1) * (E - 0.5 * rho * (v_x ** 2 + v_y ** 2))

    # Apply pressure floor
    P = np.maximum(P, 1e-8)

    # Reconstruction in x-direction
    rho_L_x, rho_R_x = reconstruct_2d(rho, theta, axis=0)
    v_x_L_x, v_x_R_x = reconstruct_2d(v_x, theta, axis=0)
    v_y_L_x, v_y_R_x = reconstruct_2d(v_y, theta, axis=0)
    P_L_x, P_R_x = reconstruct_2d(P, theta, axis=0)

    U_L_x = np.zeros_like(U)
    U_R_x = np.zeros_like(U)
    U_L_x[:, :, 0] = rho_L_x
    U_L_x[:, :, 1] = rho_L_x * v_x_L_x
    U_L_x[:, :, 2] = rho_L_x * v_y_L_x
    U_L_x[:, :, 3] = P_L_x / (gamma - 1) + 0.5 * rho_L_x * (v_x_L_x ** 2 + v_y_L_x ** 2)
    U_R_x[:, :, 0] = rho_R_x
    U_R_x[:, :, 1] = rho_R_x * v_x_R_x
    U_R_x[:, :, 2] = rho_R_x * v_y_R_x
    U_R_x[:, :, 3] = P_R_x / (gamma - 1) + 0.5 * rho_R_x * (v_x_R_x ** 2 + v_y_R_x ** 2)

    F_HLL_x = compute_hll_flux(U_L_x, U_R_x, gamma, axis=0)

    # Flux differences in x-direction
    L[2:-2, :, :] -= (F_HLL_x[2:-2, :, :] - F_HLL_x[1:-3, :, :]) / dx

    # Reconstruction in y-direction
    rho_L_y, rho_R_y = reconstruct_2d(rho, theta, axis=1)
    v_x_L_y, v_x_R_y = reconstruct_2d(v_x, theta, axis=1)
    v_y_L_y, v_y_R_y = reconstruct_2d(v_y, theta, axis=1)
    P_L_y, P_R_y = reconstruct_2d(P, theta, axis=1)

    U_L_y = np.zeros_like(U)
    U_R_y = np.zeros_like(U)
    U_L_y[:, :, 0] = rho_L_y
    U_L_y[:, :, 1] = rho_L_y * v_x_L_y
    U_L_y[:, :, 2] = rho_L_y * v_y_L_y
    U_L_y[:, :, 3] = P_L_y / (gamma - 1) + 0.5 * rho_L_y * (v_x_L_y ** 2 + v_y_L_y ** 2)
    U_R_y[:, :, 0] = rho_R_y
    U_R_y[:, :, 1] = rho_R_y * v_x_R_y
    U_R_y[:, :, 2] = rho_R_y * v_y_R_y
    U_R_y[:, :, 3] = P_R_y / (gamma - 1) + 0.5 * rho_R_y * (v_x_R_y ** 2 + v_y_R_y ** 2)

    F_HLL_y = compute_hll_flux(U_L_y, U_R_y, gamma, axis=1)

    # Flux differences in y-direction
    L[:, 2:-2, :] -= (F_HLL_y[:, 2:-2, :] - F_HLL_y[:, 1:-3, :]) / dy

    return L

def apply_boundary_conditions(U):
    """
    Apply boundary conditions to U.
    """
    nx = U.shape[0] - 4
    ny = U.shape[1] - 4
    # Periodic boundary conditions in y-direction
    U[:, 0, :] = U[:, ny + 0, :]
    U[:, 1, :] = U[:, ny + 1, :]
    U[:, ny + 2, :] = U[:, 2, :]
    U[:, ny + 3, :] = U[:, 3, :]

    # Reflective boundary conditions in x-direction
    U[0, :, :] = U[3, :, :]
    U[1, :, :] = U[2, :, :]
    U[nx + 2, :, :] = U[nx + 1, :, :]
    U[nx + 3, :, :] = U[nx, :, :]

    return U

def enforce_positivity(U, gamma):
    """
    Enforce positivity of density and pressure.
    """
    rho = U[:, :, 0]
    rho = np.maximum(rho, 1e-8)
    U[:, :, 0] = rho

    rho_vx = U[:, :, 1]
    rho_vy = U[:, :, 2]
    E = U[:, :, 3]

    v_x = rho_vx / rho
    v_y = rho_vy / rho

    E_kin = 0.5 * rho * (v_x ** 2 + v_y ** 2)
    E_int = E - E_kin
    E_int = np.maximum(E_int, 1e-8)
    E = E_int + E_kin
    U[:, :, 3] = E

    return U

def shu_osher(U, dx, dy, dt, gamma, theta):
    """
    Perform a time step using the Shu-Osher third-order scheme in 2D.
    """
    U1 = np.copy(U)
    U2 = np.copy(U)
    U_new = np.copy(U)

    # Stage 1
    L1 = compute_L(U, dx, dy, gamma, theta)
    U1 += dt * L1
    U1 = apply_boundary_conditions(U1)
    U1 = enforce_positivity(U1, gamma)
    print('stage 1 done')
    # Stage 2
    L2 = compute_L(U1, dx, dy, gamma, theta)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    U2 = apply_boundary_conditions(U2)
    U2 = enforce_positivity(U2, gamma)

    # Stage 3
    L3 = compute_L(U2, dx, dy, gamma, theta)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * L3)
    U_new = apply_boundary_conditions(U_new)
    U_new = enforce_positivity(U_new, gamma)

    return U_new

def compute_time_step(U, dx, dy, cfl, gamma):
    """
    Compute time step size dt based on CFL condition in 2D.
    """
    rho = U[:, :, 0]
    rho_vx = U[:, :, 1]
    rho_vy = U[:, :, 2]
    E = U[:, :, 3]

    # Prevent division by zero
    rho = np.maximum(rho, 1e-8)

    v_x = rho_vx / rho
    v_y = rho_vy / rho

    P = (gamma - 1) * (E - 0.5 * rho * (v_x ** 2 + v_y ** 2))

    # Apply pressure floor
    P = np.maximum(P, 1e-8)

    c_s = np.sqrt(gamma * P / rho)
    max_speed_x = np.max(np.abs(v_x) + c_s)
    max_speed_y = np.max(np.abs(v_y) + c_s)

    # Avoid division by zero or NaN
    max_speed_x = np.maximum(max_speed_x, 1e-8)
    max_speed_y = np.maximum(max_speed_y, 1e-8)

    dt_x = cfl * dx / max_speed_x
    dt_y = cfl * dy / max_speed_y

    dt = min(dt_x, dt_y)

    return dt

def initialize_2d_sod_shock_tube(nx, ny, x_length, y_length, gamma):
    """
    Initialize a 2D Sod shock tube problem with periodic boundary in y-direction.
    """
    dx = x_length / nx
    dy = y_length / ny
    x = np.linspace(-2 * dx, x_length + 2 * dx, nx + 4)  # Include ghost cells
    y = np.linspace(-2 * dy, y_length + 2 * dy, ny + 4)
    U = np.zeros((nx + 4, ny + 4, 4))  # [rho, rho*v_x, rho*v_y, E]

    # Initial conditions: Sod shock tube along x-direction
    rho_L = 1.0
    rho_R = 0.125
    P_L = 1.0
    P_R = 0.1
    v_x_L = 0.0
    v_x_R = 0.0
    v_y = 0.0  # No initial velocity in y-direction

    # Physical domain indices
    i_start = 2
    i_end = nx + 2
    j_start = 2
    j_end = ny + 2

    # Set initial conditions
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            if x[i] < x_length / 2:
                rho = rho_L
                P = P_L
                v_x = v_x_L
            else:
                rho = rho_R
                P = P_R
                v_x = v_x_R
            E = P / (gamma - 1) + 0.5 * rho * (v_x ** 2 + v_y ** 2)
            U[i, j, :] = [rho, rho * v_x, rho * v_y, E]

    return x, y, U

def main():
    start_time = time.time()

    gamma = 1.4
    nx = 200  # Number of grid points in x-direction
    ny = 100  # Number of grid points in y-direction
    x_length = 1.0
    y_length = 0.5
    dx = x_length / nx
    dy = y_length / ny

    x, y, U = initialize_2d_sod_shock_tube(nx, ny, x_length, y_length, gamma)

    # Apply boundary conditions
    U = apply_boundary_conditions(U)

    # Time parameters
    t = 0.0
    t_final = 0.15
    cfl = 0.5
    theta = 1.5

    # Time evolution loop
    while t < t_final:
        dt = compute_time_step(U, dx, dy, cfl, gamma)
        if not np.isfinite(dt) or dt <= 0.0:
            print("Non-finite or non-positive time step encountered.")
            break
        if t + dt > t_final:
            dt = t_final - t
        U = shu_osher(U, dx, dy, dt, gamma, theta)
        t += dt
        print(f"t = {t:.4f}, dt = {dt:.4e}")
        if not np.all(np.isfinite(U)):
            print("Non-finite values encountered in U.")
            # print(np.argwhere(~np.isfinite(U)))
            break

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    # Extract variables for plotting (physical domain)
    i_start = 2
    i_end = nx + 2
    j_start = 2
    j_end = ny + 2

    x_physical = x[i_start:i_end]
    y_physical = y[j_start:j_end]
    X, Y = np.meshgrid(x_physical, y_physical, indexing='ij')

    rho = U[i_start:i_end, j_start:j_end, 0]
    rho_vx = U[i_start:i_end, j_start:j_end, 1]
    rho_vy = U[i_start:i_end, j_start:j_end, 2]
    E = U[i_start:i_end, j_start:j_end, 3]

    # Prevent division by zero
    rho = np.maximum(rho, 1e-8)
    v_x = rho_vx / rho
    v_y = rho_vy / rho
    P = (gamma - 1) * (E - 0.5 * rho * (v_x ** 2 + v_y ** 2))
    P = np.maximum(P, 1e-8)

    # Plot density contour
    plt.figure(figsize=(8, 4))
    plt.contourf(X, Y, rho, levels=50, cmap='jet')
    plt.colorbar(label='Density')
    plt.title('Density Contour at t = {:.2f}'.format(t))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    # plt.show()

    # Plot velocity field
    plt.figure(figsize=(8, 4))
    plt.quiver(X[::5, ::5], Y[::5, ::5], v_x[::5, ::5], v_y[::5, ::5])
    plt.title('Velocity Field at t = {:.2f}'.format(t))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    # plt.show()

    # Plot pressure contour
    plt.figure(figsize=(8, 4))
    plt.contourf(X, Y, P, levels=50, cmap='jet')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Contour at t = {:.2f}'.format(t))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
