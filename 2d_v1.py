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
    min_abs = np.min([np.abs(a), np.abs(b), np.abs(c)], axis=0)
    return 0.25 * np.abs(sgn_a + sgn_b) * (sgn_a + sgn_c) * min_abs

def deconstruct_U(U, gamma=1.4):
    """
    Get rho, v, and p from U
    """
    rho = U[:, :, 0]
    vx = U[:, :, 1] / U[:, :, 0]
    vy = U[:, :, 2] / U[:, :, 0]
    P = (gamma - 1) * (U[:, :, 3] - 0.5 * (np.square(U[:, :, 1]) + np.square(U[:, :, 2])) / U[:, :, 0])
    return rho, vx, vy, P

def compute_F_and_G(U, gamma):
    """
    Compute the flux vector F given conserved variables U.
    """
    rho, vx, vy, P = deconstruct_U(U, gamma)
    E = U[:, 3]
    F = np.zeros_like(U)
    F[:, 0] = rho * vx
    F[:, 1] = rho * np.square(vx) + P
    F[:, 2] = rho * vx * vy
    F[:, 3] = (E + P) * vx

    G = np.zeros_like(U)
    G[:, 0] = rho * vy
    G[:, 1] = rho * vx * vy
    G[:, 2] = rho * np.square(vy) + P
    G[:, 3] = (E + P) * vy

    return F, G

def apply_boundary_conditions(U):
    """
    Apply periodic boundary conditions in both directions.
    """
    U[0, :] = U[-2, :]
    U[-1, :] = U[1, :]
    U[:, 0] = U[:, -2]
    U[:, -1] = U[:, 1]
    return U

def reconstruct(c, theta):
    """
    Reconstruct left and right states at cell interfaces using minmod limiter.
    """
    nx, ny = c.shape
    c_L = np.zeros((nx + 1, ny + 1))
    c_R = np.zeros((nx + 1, ny + 1))
    sigma = np.zeros((nx, ny))

    # Compute slopes
    for i in range(2, nx - 2):
        for j in range(2, ny - 2):
            delta_c_minus = c[i, j] - c[i - 1, j]
            delta_c_plus = c[i + 1, j] - c[i, j]
            delta_c_center = 0.5 * (c[i + 1, j] - c[i - 1, j])
            sigma[i, j] = minmod(theta * delta_c_minus, delta_c_center, theta * delta_c_plus)

    # # Reconstruct left and right states at interfaces
    # for i in range(2, nx - 2):
    #     for j in range(2, ny - 2):
    #         c_L[i, j] = c[i, j] + 0.5 * sigma[i, j]
    #         c_R[i, j] = c[i + 1, j] - 0.5 * sigma[i + 1, j]

    c_L[2:nx-2, 2:ny-2] = c[2:nx-2, 2:ny-2] + 0.5 * sigma[2:nx-2, 2:ny-2]
    c_R[2:nx-2, 2:ny-2] = c[3:nx-1, 2:ny-2] - 0.5 * sigma[3:nx-1, 2:ny-2]

    return c_L, c_R

def compute_flux(U, gamma):
    """
    Compute the flux vector F given conserved variables U in 2D.
    """
    rho = U[0]
    rho_vx = U[1]
    rho_vy = U[2]
    E = U[3]
    
    vx = rho_vx / rho
    vy = rho_vy / rho
    P = (gamma - 1) * (E - 0.5 * rho * (vx ** 2 + vy ** 2))
    
    Fx = np.zeros(4)
    Fx[0] = rho * vx
    Fx[1] = rho * vx ** 2 + P
    Fx[2] = rho * vx * vy
    Fx[3] = (E + P) * vx
    
    Fy = np.zeros(4)
    Fy[0] = rho * vy
    Fy[1] = rho * vx * vy
    Fy[2] = rho * vy ** 2 + P
    Fy[3] = (E + P) * vy
    
    return Fx, Fy

def compute_hll_flux(U_L, U_R, gamma):
    """
    Compute HLL flux at an interface given left and right states in 2D.
    """
    rho_L = U_L[0]
    rho_R = U_R[0]
    vx_L = U_L[1] / rho_L
    vx_R = U_R[1] / rho_R
    vy_L = U_L[2] / rho_L
    vy_R = U_R[2] / rho_R
    E_L = U_L[3]
    E_R = U_R[3]
    
    P_L = (gamma - 1) * (E_L - 0.5 * rho_L * (vx_L ** 2 + vy_L ** 2))
    P_R = (gamma - 1) * (E_R - 0.5 * rho_R * (vx_R ** 2 + vy_R ** 2))

    P_L = np.maximum(P_L, 1e-6)
    P_R = np.maximum(P_R, 1e-6)
    rho_L = np.maximum(rho_L, 1e-6)
    rho_R = np.maximum(rho_R, 1e-6)

    c_Lx = np.sqrt(gamma * P_L / rho_L)
    c_Ly = np.sqrt(gamma * P_L / rho_L)
    c_Rx = np.sqrt(gamma * P_R / rho_R)
    c_Ry = np.sqrt(gamma * P_R / rho_R)

    S_Lx = min(vx_L - c_Lx, vx_R - c_Rx, 0.0)
    S_Rx = max(vx_L + c_Lx, vx_R + c_Rx, 0.0)
    S_Ly = min(vy_L - c_Ly, vy_R - c_Ry, 0.0)
    S_Ry = max(vy_L + c_Ly, vy_R + c_Ry, 0.0)
    
    Fx_L, Fy_L = compute_flux(U_L, gamma)
    Fx_R, Fy_R = compute_flux(U_R, gamma)
    
    # HLL Flux calculation in 2D
    if S_Lx >= 0 and S_Ly >= 0:
        F_HLLx = Fx_L
        F_HLLy = Fy_L
    elif S_Rx <= 0 and S_Ry <= 0:
        F_HLLx = Fx_R
        F_HLLy = Fy_R
    else:
        F_HLLx = (S_Rx * Fx_L - S_Lx * Fx_R + S_Lx * S_Rx * (U_R - U_L)) / (S_Rx - S_Lx)
        F_HLLy = (S_Ry * Fy_L - S_Ly * Fy_R + S_Ly * S_Ry * (U_R - U_L)) / (S_Ry - S_Ly)

    return F_HLLx, F_HLLy

def compute_L(U, nx, ny, dx, dy, gamma, theta):
    """
    Compute the flux residuals in the Shu-Osher scheme in 2D using the HLL flux.
    """
    Lx = np.zeros_like(U)
    Ly = np.zeros_like(U)

    # Iterate over all cells (excluding the boundary cells)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Left and right states for x direction
            U_Lx = U[i, j, :]
            U_Rx = U[i + 1, j, :]

            # Left and right states for y direction
            U_Ly = U[i, j, :]
            U_Ry = U[i, j + 1, :]

            # Compute the fluxes using the HLL method
            Fx_L, Fy_L = compute_hll_flux(U_Lx, U_Rx, gamma)
            Fx_R, Fy_R = compute_hll_flux(U_Ly, U_Ry, gamma)

            # Update Lx and Ly for this interface
            Lx[i, j, :] = (Fx_R - Fx_L) / dx
            Ly[i, j, :] = (Fy_R - Fy_L) / dy

    return Lx, Ly

def shu_osher(U, nx, ny, dx, dy, dt, gamma, theta):
    """
    Perform a time step using the Shu-Osher third-order scheme in 2D.
    """
    # Stage 1
    L1x, L1y = compute_L(U, nx, ny, dx, dy, gamma, theta)
    U1 = U + dt * L1x + dt * L1y
    U1 = apply_boundary_conditions(U1)

    # Stage 2
    L2x, L2y = compute_L(U1, nx, ny, dx, dy, gamma, theta)
    U2 = 0.75 * U + 0.25 * (U1 + dt * (L2x + L2y))
    U2 = apply_boundary_conditions(U2)

    # Stage 3
    L3x, L3y = compute_L(U2, nx, ny, dx, dy, gamma, theta)
    U_new = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * (L3x + L3y))
    U_new = apply_boundary_conditions(U_new)

    return U_new

def compute_time_step(U, dx, dy, cfl, gamma):
    """
    Compute time step size dt based on CFL condition in 2D.
    """
    rho = U[:, :, 0]
    rho_vx = U[:, :, 1]
    rho_vy = U[:, :, 2]
    E = U[:, :, 3]
    
    vx = rho_vx / rho
    vy = rho_vy / rho
    P = (gamma - 1) * (E - 0.5 * rho * (vx ** 2 + vy ** 2))

    P = np.maximum(P, 1e-6)
    rho = np.maximum(rho, 1e-6)

    c_s = np.sqrt(gamma * P / rho)
    max_speed_x = np.max(np.abs(vx) + c_s)
    max_speed_y = np.max(np.abs(vy) + c_s)
    dt = cfl * min(dx, dy) / max(max_speed_x, max_speed_y)
    return dt

def save_2d_plot(data, title, xlabel, ylabel, filename):
    """Save 2D plot as a PNG file."""
    plt.figure(figsize=(12, 8))
    plt.imshow(data.T, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def save_3d_plot(data, title, xlabel, ylabel, zlabel, filename):
    """Save 3D surface plot as a PNG file."""
    x = np.linspace(0, 1, data.shape[0])
    y = np.linspace(0, 1, data.shape[1])
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data.T, cmap='viridis', edgecolor='none')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(filename)
    plt.close()

def main():
    start_time = time.time()

    gamma = 1.4
    nx = 200
    ny = 200
    x_start = 0.0
    x_end = 1.0
    y_start = 0.0
    y_end = 1.0
    dx = (x_end - x_start) / (nx - 1)
    dy = (y_end - y_start) / (ny - 1)
    x = np.linspace(x_start, x_end, nx)
    y = np.linspace(y_start, y_end, ny)

    # Initialize variables
    rho = np.zeros((nx, ny))
    vx = np.zeros((nx, ny))
    vy = np.zeros((nx, ny))
    P = np.zeros((nx, ny))

    # Initial conditions for 2D Sod shock tube
    rho_L = 10.0
    rho_R = 1.0
    P_L = 8.0
    P_R = 1.0
    vx_L = 0.0
    vx_R = 0.0
    vy_L = 0.0
    vy_R = 0.0

    for i in range(nx):
        for j in range(ny):
            if x[i] <= 0.5:
                rho[i, j] = rho_L
                P[i, j] = P_L
                vx[i, j] = vx_L
                vy[i, j] = vy_L
            else:
                rho[i, j] = rho_R
                P[i, j] = P_R
                vx[i, j] = vx_R
                vy[i, j] = vy_R

    E = P / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)
    U = np.zeros((nx, ny, 4))
    U[:, :, 0] = rho
    U[:, :, 1] = rho * vx
    U[:, :, 2] = rho * vy
    U[:, :, 3] = E

    # Time parameters
    t = 0.0
    t_final = 0.25
    cfl = 0.5
    theta = 1.5

    # Time evolution loop
    while t < t_final:
        dt = compute_time_step(U, dx, dy, cfl, gamma)
        if t + dt > t_final:
            dt = t_final - t
        U = shu_osher(U, nx, ny, dx, dy, dt, gamma, theta)
        t += dt
        print(f"t = {t:.4f}, dt = {dt:.4e}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    # Debugging step: Ensure each plot is being saved
    try:
        # Save 2D plots for density, pressure, and velocity
        print("Saving 2D plots...")
        save_2d_plot(U[:, :, 0], 'Density', 'x', 'y', 'density.png')
        save_2d_plot(U[:, :, 3], 'Pressure', 'x', 'y', 'pressure.png')
        save_2d_plot(U[:, :, 1], 'Velocity X', 'x', 'y', 'velocity_x.png')
        save_2d_plot(U[:, :, 2], 'Velocity Y', 'x', 'y', 'velocity_y.png')

        # Save 3D surface plots for density, pressure, and velocity
        print("Saving 3D plots...")
        save_3d_plot(U[:, :, 0], 'Density 3D', 'x', 'y', 'Density', 'density_3d.png')
        save_3d_plot(U[:, :, 3], 'Pressure 3D', 'x', 'y', 'Pressure', 'pressure_3d.png')
        save_3d_plot(U[:, :, 1], 'Velocity X 3D', 'x', 'y', 'Velocity X', 'velocity_x_3d.png')
        save_3d_plot(U[:, :, 2], 'Velocity Y 3D', 'x', 'y', 'Velocity Y', 'velocity_y_3d.png')

    except Exception as e:
        print(f"Error while saving plots: {e}")
    
    print("Plotting complete.")

if __name__ == "__main__":
    main()