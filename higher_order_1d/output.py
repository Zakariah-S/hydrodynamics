import numpy as np
import matplotlib.pyplot as plt

def plot_from_one_U(U_i, x, gamma=1.4):
    # Extract variables for plotting
    rho = U_i[:, 0]
    rho_v = U_i[:, 1]
    E = U_i[:, 2]
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

def plot_one_time(x, rho, v, p, t=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(x, rho)
    if t: plt.title(f'Sod Shock Tube Results at Time {np.round(t, 2)} s')
    else: plt.title('Sod Shock Tube Results')
    plt.ylabel('Density')

    plt.subplot(3, 1, 2)
    plt.plot(x, v, color='green')
    plt.ylabel('Velocity')

    plt.subplot(3, 1, 3)
    plt.plot(x, p, color='red')
    plt.xlabel('Position x')
    plt.ylabel('Pressure')

    plt.tight_layout()
    plt.show()

def save_data(savename, U, x, t, gamma=1.4):
    # Extract variables for plotting
    rho = U[:, :, 0]
    rho_v = U[:, :, 1]
    v = rho_v / rho
    p = (gamma - 1) * (U[:, :, 2] - 0.5 * rho * v ** 2)

    # Exclude ghost cells
    x = x[2:-2]
    rho = rho[:, 2:-2]
    v = v[:, 2:-2]
    p = p[:, 2:-2]

    np.savez_compressed(savename, t=t, x=x, rho=rho, v=v, p=p)

def load_data(infile):
    loaded = np.load(infile, mmap_mode='r')
    t = loaded['t']
    x = loaded['x']
    rho = loaded['rho']
    v = loaded['v']
    p = loaded['p']
    return t, x, rho, v, p

def animate(t, x, rho, v, p):
    import matplotlib.animation as mani
    # Plot results
    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(3, 1, 1)
    density, = plt.plot(x, rho[0])
    plt.xlim(0., 1.)
    plt.ylim(np.min(rho) - 0.5, np.max(rho) + 0.5)
    plt.title('Sod Shock Tube Results')
    plt.ylabel('Density')
    time_text = plt.annotate('t = 0 s', xy=(0.93, 0.9), xycoords='axes fraction', xytext=(0., 2.), textcoords='offset fontsize')

    fig.add_subplot(3, 1, 2)
    vel, = plt.plot(x, v[0], color='green')
    plt.xlim(0., 1.)
    plt.ylim(np.min(v) - 0.5, np.max(v) + 0.5)
    plt.ylabel('Velocity')

    fig.add_subplot(3, 1, 3)
    pressure, = plt.plot(x, p[0], color='red')
    plt.xlim(0., 1.)
    plt.ylim(np.min(p) - 0.5, np.max(p) + 0.5)
    plt.xlabel('Position x')
    plt.ylabel('Pressure')

    plt.tight_layout()

    def update(frame):
        density.set_data(x, rho[frame])
        vel.set_data(x, v[frame])
        pressure.set_data(x, p[frame])
        time_text.set_text(f't = {np.round(t[frame], 2):.2f} s')
        return density, vel, pressure, time_text

    ani = mani.FuncAnimation(fig=fig, func=update, frames=range(1, t.size), interval=10)

    plt.show()