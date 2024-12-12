import numpy as np
import matplotlib.pyplot as plt
import time  # Importing the time module

# Initialization Module
def initialize_sod_shock_tube(nx, x_length, gamma):
    x = np.linspace(0, x_length, nx)
    U = np.zeros((nx, 3))  # [rho, rho*v, E]
    
    # Initial conditions
    x_mid = x_length / 2
    for i in range(nx):
        if x[i] < x_mid:
            rho = 1.0       # Left density
            P = 1.0         # Left pressure
        else:
            rho = 0.1       # Right density
            P = 0.125       # Right pressure
        v = 0.0             # Initial velocity
        E = P / (gamma - 1) + 0.5 * rho * v ** 2
        U[i, :] = [rho, rho * v, E]
    
    return x, U

# Evolution Module
def compute_flux(U, gamma):
    rho = U[0]
    rho_v = U[1]
    E = U[2]
    v = rho_v / rho
    P = (gamma - 1) * (E - 0.5 * rho * v ** 2)
    F = np.array([rho_v,
                  rho_v * v + P,
                  (E + P) * v])
    return F

def compute_hll_flux(U_L, U_R, gamma):
    # Compute speeds
    rho_L, rho_R = U_L[0], U_R[0]
    v_L, v_R = U_L[1] / rho_L, U_R[1] / rho_R
    P_L = (gamma - 1) * (U_L[2] - 0.5 * rho_L * v_L ** 2)
    P_R = (gamma - 1) * (U_R[2] - 0.5 * rho_R * v_R ** 2)
    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)
    
    lambda_L = min(v_L - c_L, v_R - c_R, 0)
    lambda_R = max(v_L + c_L, v_R + c_R, 0)

    # Compute HLL flux
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)
    if lambda_R != lambda_L:
        flux = (lambda_R * F_L - lambda_L * F_R + lambda_R * lambda_L * (U_R - U_L)) / (lambda_R - lambda_L)
    else:
        flux = F_L  # Avoid division by zero if speeds are equal
    
    return flux

def evolve(U, nx, dx, dt, gamma):
    U_new = U.copy()
    
    # Compute fluxes at interfaces
    for i in range(1, nx - 1):
        U_L = U[i - 1, :]
        U_R = U[i, :]
        
        # HLL flux
        flux_L = compute_hll_flux(U_L, U_R, gamma)
        flux_R = compute_hll_flux(U[i, :], U[i + 1, :], gamma)
        
        # Update conserved variables
        U_new[i, :] = U_new[i, :] - dt / dx * (flux_R - flux_L)
    
    return U_new

def compute_time_step(dx, max_wave_speed, cfl_number):
    return cfl_number * dx / max_wave_speed

# Main Script
# Parameters
nx = 500
x_length = 1.0
gamma = 1.4
cfl_number = 0.9
t_final = 0.25
save_interval = 0.05  # Save an image every this time interval
nt = 50  # Number of time steps for saving data

# Initialize
x, U = initialize_sod_shock_tube(nx, x_length, gamma)
dx = x[1] - x[0]
t = 0.0
time_steps = 0

# Start the simulation time tracking
start_time = time.time()  # Record the start time

# Lists to store multiple time traces for plotting
rho_traces = []
v_traces = []
P_traces = []
time_stamps = []  # List to store actual time steps for labels

# Time evolution
while t < t_final:
    # Compute primitive variables
    rho = U[:, 0]
    rho_v = U[:, 1]
    E = U[:, 2]
    v = rho_v / rho
    P = (gamma - 1) * (E - 0.5 * rho * v ** 2)
    c = np.sqrt(gamma * P / rho)
    max_wave_speed = np.max(np.abs(v) + c)
    
    # Compute time step
    dt = compute_time_step(dx, max_wave_speed, cfl_number)
    if t + dt > t_final:
        dt = t_final - t
    
    # Evolve
    U = evolve(U, nx, dx, dt, gamma)
    t += dt
    time_steps += 1
    print(f"Time: {t:.5f}, Time step: {dt:.5f}")
    
    # Save plot at specified intervals
    if time_steps % nt == 0:  # Save every nt steps (e.g., 50 steps)
        rho_traces.append(U[:, 0])  # Save density
        v_traces.append(U[:, 1] / U[:, 0])  # Save velocity
        P_traces.append((gamma - 1) * (U[:, 2] - 0.5 * U[:, 0] * (U[:, 1] / U[:, 0]) ** 2))  # Save pressure
        
        # Record the actual simulation time
        current_time = time.time() - start_time  # Calculate the elapsed time since start
        time_stamps.append(current_time)

# Plotting function
def plot():
    plt.figure(figsize=(10, 8))

    # Density plot with multiple traces
    plt.subplot(3, 1, 1)
    for i, rho_trace in enumerate(rho_traces):
        plt.plot(x, rho_trace, label=f'Time {time_stamps[i]:.2f}s')  # Use actual time for label
    plt.ylabel('Density', size = 17)
    plt.yticks(size = 16)
    plt.xticks(size = 16)
    plt.legend()  # Ensure that the legend is added after all the lines are plotted

    # Velocity plot with multiple traces
    plt.subplot(3, 1, 2)
    for i, v_trace in enumerate(v_traces):
        plt.plot(x, v_trace, label=f'Time {time_stamps[i]:.2f}s')  # Use actual time for label
    plt.ylabel('Velocity', size = 17)
    plt.yticks(size = 16)
    plt.xticks(size = 16)
    plt.legend()  # Ensure that the legend is added after all the lines are plotted

    # Pressure plot with multiple traces
    plt.subplot(3, 1, 3)
    for i, P_trace in enumerate(P_traces):
        plt.plot(x, P_trace, label=f'Time {time_stamps[i]:.2f}s')  # Use actual time for label
    plt.ylabel('Pressure',size = 17 )
    plt.xlabel('Position', size = 17)
    plt.yticks(size = 16)
    plt.xticks(size = 16)
    plt.legend()  # Ensure that the legend is added after all the lines are plotted

    plt.tight_layout()
    plt.savefig('shock_tube_solution_multiple_traces_with_time.png', dpi=300)
    plt.show()

plot()
