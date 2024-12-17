"""
Module with functions that save, load, plot, and animate data generated by the 2D sim.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_one_time(infile, frame):
    t, x, rho, v, p = load_data('sodshock200.npz')
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(x, rho[frame])
    plt.title(f'Sod Shock Tube Results at Time {np.round(t[frame], 2)} s')
    plt.ylabel('Density')

    plt.subplot(3, 1, 2)
    plt.plot(x, v[frame], color='green')
    plt.ylabel('Velocity')

    plt.subplot(3, 1, 3)
    plt.plot(x, p[frame], color='red')
    plt.xlabel('Position x')
    plt.ylabel('Pressure')

    plt.tight_layout()
    plt.show()

def save_data(savename, U, x, t, gamma=1.4):
    # Extract variables for plotting
    rho = U[:, 0]
    rho_v = U[:, 1]
    v = rho_v / rho
    p = (gamma - 1) * (U[:, 2] - 0.5 * rho * v ** 2)

    np.savez_compressed(savename, t=t, x=x, rho=rho, v=v, p=p)

def load_data(infile):
    loaded = np.load(infile, mmap_mode='r')
    t = loaded['t']
    x = loaded['x']
    rho = loaded['rho']
    v = loaded['v']
    p = loaded['p']
    return t, x, rho, v, p

def animate_from_file(infile, savename, title='Sod Shock Simulation Results', interval=50, fps=15):
    import matplotlib.animation as mani
    t, x, rho, v, p = load_data(infile)

    # Plot results
    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(3, 1, 1)
    density, = plt.plot(x, rho[0])
    plt.xlim(x[0], x[-1])
    height = np.max(rho) - np.min(rho)
    plt.ylim(np.min(rho) - 0.5*height, np.max(rho) + 0.5*height)
    plt.title(title)
    plt.ylabel('Density')
    time_text = plt.annotate('t = 0 s', xy=(0.93, 0.9), xycoords='axes fraction', xytext=(0., 2.), textcoords='offset fontsize')

    fig.add_subplot(3, 1, 2)
    vel, = plt.plot(x, v[0], color='green')
    plt.xlim(x[0], x[-1])
    height = np.max(v) - np.min(v)
    plt.ylim(np.min(v) - 0.5*height, np.max(v) + 0.5*height)
    plt.ylabel('Velocity')

    fig.add_subplot(3, 1, 3)
    pressure, = plt.plot(x, p[0], color='red')
    plt.xlim(x[0], x[-1])
    height = np.max(p) - np.min(p)
    plt.ylim(np.min(p) - 0.5*height, np.max(p) + 0.5*height)
    plt.xlabel('Position x')
    plt.ylabel('Pressure')

    plt.tight_layout()

    def update(frame):
        density.set_data(x, rho[frame])
        vel.set_data(x, v[frame])
        pressure.set_data(x, p[frame])
        time_text.set_text(f't = {np.round(t[frame], 2):.2f} s')
        return density, vel, pressure, time_text

    if savename:
        ani = mani.FuncAnimation(fig=fig, func=update, frames=range(1, t.size), interval=interval)

        writer = mani.PillowWriter(fps=fps,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save(savename, writer=writer)

    plt.show()

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

def avg_resid_over_time(infile1, infile2):
    t, x, rho1, v1, p1 = load_data(infile1)
    t2, x2, rho2, v2, p2 = load_data(infile2)

    assert np.all(np.abs(t - t2) < 0.01)
    assert np.all(np.abs(x - x2) < 0.001)
    assert rho1.shape == rho2.shape
    assert v1.shape == v2.shape
    assert p1.shape == p2.shape

    rho_mean_resid = np.mean(np.abs(rho1 - rho2), axis=1)
    v_mean_resid = np.mean(np.abs(v1 - v2), axis=1)
    p_mean_resid = np.mean(np.abs(p1 - p2), axis=1)

    return t, rho_mean_resid, v_mean_resid, p_mean_resid