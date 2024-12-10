import numpy as np
import matplotlib.pyplot as plt

def plot_data(U, x, gamma=1.4):
    # Extract variables for plotting
    rho = U[:, 0]
    rho_v = U[:, 1]
    E = U[:, 2]
    v = rho_v / rho
    P = (gamma - 1) * (E - 0.5 * rho * v ** 2)

    # Exclude ghost cells for plotting
    x_plot = x[2:-2]
    rho_plot = rho[2:-2]
    v_plot = v[2:-2]
    P_plot = P[2:-2]

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(x_plot, rho_plot, label='Density')
    plt.title('Sod Shock Tube Results')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(x_plot, v_plot, label='Velocity', color='green')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(x_plot, P_plot, label='Pressure', color='red')
    plt.xlabel('Position x')
    plt.ylabel('Pressure')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_data():
    pass