"""
Module with functions that save, load, plot, and animate data generated by the 1D sims.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_3D(x, y, data, title="", xlabel="", ylabel="", zlabel=""):
    """Save 3D surface plot as a PNG file."""
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    plt.show()
    plt.close()

def save_data(savename, U, x, y, t, gamma=1.4):
    # Extract variables for plotting
    U = U[:, 2:-2, 2:-2, :]
    rho = U[:, :, :, 0]
    vx = U[:, :, :, 1] / U[:, :, :, 0]
    vy = U[:, :, :, 2] / U[:, :, :, 0]
    p = (gamma - 1) * (U[:, :, :, 2] - 0.5 * rho * (np.square(vx) + np.square(vy)))

    # Exclude ghost cells
    x = x[2:-2]
    y = y[2:-2]

    np.savez_compressed(savename, t=t, x=x, y=y, rho=rho, vx=vx, vy=vy, p=p)

def load_data(infile):
    loaded = np.load(infile, mmap_mode='r')
    t = loaded['t']
    x = loaded['x']
    y = loaded['y']
    rho = loaded['rho']
    vx = loaded['vx']
    vy = loaded['vy']
    p = loaded['p']
    return t, x, y, rho, vx, vy, p

def animate_3D(t, x, y, rho, vx, vy, p, title='Sod Shock Simulation Results', interval=50):
    import matplotlib.animation as mani
    # Plot results
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for val in [rho]:
        data = ax.plot_surface(X, Y, val[0], cmap='viridis', edgecolor='none')
        ax.axes.set_xlim3d(x[0], x[-1]) 
        ax.axes.set_ylim3d(y[0], y[-1]) 
        ax.axes.set_zlim3d(-10., 10.)
        plt.title(title)
        ax.set_zlabel('Density')
        time_text = plt.annotate('t = 0 s', xy=(0.93, 0.9), xycoords='axes fraction', xytext=(0., 2.), textcoords='offset fontsize')

        def update(frame):
            ax.clear()
            data = ax.plot_surface(X, Y, val[frame], cmap='viridis', edgecolor='none')
            time_text = plt.annotate(f't = {np.round(t[frame], 2):.2f} s', xy=(0.93, 0.9), xycoords='axes fraction', xytext=(0., 2.), textcoords='offset fontsize')

            return data

        ani = mani.FuncAnimation(fig=fig, func=update, frames=range(1, t.size), interval=interval)

        plt.show()

def animate_from_file(infile, title='Sod Shock Simulation Results', interval=50):
    t, x, y, rho, vx, vy, p = load_data(infile)
    animate_3D(t, x, y, rho, vx, vy, p, title, interval=interval)

def residuals_animation(infile1, infile2, title='Sod Shock Simulation Residuals', legend1='1', legend2='2'):
    import matplotlib.animation as mani

    t, x, rho1, v1, p1 = load_data(infile1)
    t2, x2, rho2, v2, p2 = load_data(infile2)

    assert np.all(np.abs(t - t2) < 0.01)
    assert np.all(np.abs(x - x2) < 0.001)
    assert rho1.shape == rho2.shape
    assert v1.shape == v2.shape
    assert p1.shape == p2.shape

    fig = plt.figure(figsize=(12, 6))

    fig.add_subplot(2, 3, 1)
    dens1, = plt.plot(x, rho1[0], label=legend1)
    dens2, = plt.plot(x, rho2[0], label=legend2)
    plt.xlim(x[0], x[-1])
    plt.ylim(np.min(rho1) - 0.5, np.max(rho1) + 0.5)
    plt.ylabel('Density')
    plt.legend()
    plt.xticks(c='white')

    fig.add_subplot(2, 3, 2)
    plt.title(title)
    vel1, = plt.plot(x, v1[0], label=legend1)
    vel2, = plt.plot(x, v2[0], label=legend2)
    plt.xlim(x[0], x[-1])
    plt.ylim(np.min(v1) - 0.5, np.max(v1) + 0.5)
    plt.ylabel('Velocity')
    plt.legend()
    plt.xticks(c='white')


    fig.add_subplot(2, 3, 3)
    pres1, = plt.plot(x, p1[0], label=legend1)
    pres2, = plt.plot(x, p2[0], label=legend2)
    plt.xlim(x[0], x[-1])
    plt.ylim(np.min(p1) - 0.5, np.max(p1) + 0.5)
    plt.ylabel('Pressure')
    plt.legend()
    plt.xticks(c='white')

    time_text = plt.annotate('t = 0 s', xy=(0.77, 0.92), xycoords='axes fraction', xytext=(0., 2.), textcoords='offset fontsize')

    fig.add_subplot(2, 3, 4)
    dens_resid, = plt.plot(x, rho1[0] - rho2[0])
    plt.xlim(x[0], x[-1])
    plt.ylim(np.min(rho1 - rho2) - 0.5, np.max(rho1 - rho2) + 0.5)
    plt.xlabel('x')
    plt.ylabel('Density Residuals')

    fig.add_subplot(2, 3, 5)
    vel_resid, = plt.plot(x, v1[0] - v2[0])
    plt.xlim(x[0], x[-1])
    plt.ylim(np.min(v1 - v2) - 0.5, np.max(v1 - v2) + 0.5)
    plt.xlabel('x')
    plt.ylabel('Velocity Residuals')

    fig.add_subplot(2, 3, 6)
    pres_resid, = plt.plot(x, p1[0] - p2[0])
    plt.xlim(x[0], x[-1])
    plt.ylim(np.min(p1 - p2) - 0.5, np.max(p1 - p2) + 0.5)
    plt.xlabel('x')
    plt.ylabel('Pressure Residuals')

    plt.tight_layout()

    def update(frame):
        dens1.set_data(x, rho1[frame])
        dens2.set_data(x, rho2[frame])
        dens_resid.set_data(x, rho1[frame] - rho2[frame])
        vel1.set_data(x, v1[frame])
        vel2.set_data(x, v2[frame])
        vel_resid.set_data(x, v1[frame] - v2[frame])
        pres1.set_data(x, p1[frame])
        pres2.set_data(x, p2[frame])
        pres_resid.set_data(x, p1[frame] - p2[frame])
        time_text.set_text(f't = {np.round(t[frame], 2):.2f} s')

        return dens1, dens2, dens_resid, vel1, vel2, vel_resid, pres1, pres2, pres_resid, time_text

    ani = mani.FuncAnimation(fig=fig, func=update, frames=range(1, t.size), interval=50)
    plt.show()

def compare_files(compare_file, test_file, eps=1e-10):
    #Check if data files are the same with a very small margin of error
    t, x, rho, v, p = load_data(compare_file)
    tt, xt, rhot, vt, pt = load_data(test_file)
    print(np.all(np.abs(rho - rhot) < eps))
    print(np.all(np.abs(v - vt) < eps))
    print(np.all(np.abs(p - pt) < eps))