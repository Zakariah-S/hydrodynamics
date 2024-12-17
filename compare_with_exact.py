"""
Compare simulation results with the exact solution.
"""
import numpy as np
import matplotlib.pyplot as plt
from higher_order_1d.output import avg_resid_over_time, residuals_animation

#-----Animate residuals over 200 grid points-----#
# residuals_animation("higher_order_1d/testsodshock200.npz", 'exactsodshock1d/exactsodshock200.npz', legend1='Simulation', legend2='Exact')
# residuals_animation("lower_order_1d/sodshock200.npz", 'exactsodshock1d/exactsodshock200.npz', savename= "Animations/lower_order_compare_200.gif", legend1='Simulation', legend2='Exact')

#-----Animate residuals over 400 grid points-----#
# residuals_animation("higher_order_1d/testsodshock400.npz", 'exactsodshock1d/exactsodshock400.npz', legend1='Simulation', legend2='Exact')
# residuals_animation("lower_order_1d/sodshock400.npz", 'exactsodshock1d/exactsodshock400.npz', savename= "Animations/lower_order_compare_400.gif", legend1='Simulation', legend2='Exact')

#-----Animate residuals over 800 grid points-----#
# residuals_animation("higher_order_1d/testsodshock800.npz", 'exactsodshock1d/exactsodshock800.npz', legend1='Simulation', legend2='Exact')
# residuals_animation("lower_order_1d/sodshock800.npz", 'exactsodshock1d/exactsodshock800.npz', savename= "Animations/lower_order_compare_800.gif", legend1='Simulation', legend2='Exact')

#-----Plot avg residuals over x-----#
#Plot lower_order
directory = "lower_order_1d/"
t, r200, v200, p200 = avg_resid_over_time(directory + "sodshock200.npz", "exactsodshock1d/exactsodshock200.npz")
t, r400, v400, p400 = avg_resid_over_time(directory + "sodshock400.npz", "exactsodshock1d/exactsodshock400.npz")
t, r800, v800, p800 = avg_resid_over_time(directory + "sodshock800.npz", "exactsodshock1d/exactsodshock800.npz")

datasets_lower = [[r200, r400, r800], [v200, v400, v800], [p200, p400, p800]]

directory = "higher_order_1d/"
t, r200, v200, p200 = avg_resid_over_time(directory + "sodshock200.npz", "exactsodshock1d/exactsodshock200.npz")
t, r400, v400, p400 = avg_resid_over_time(directory + "sodshock400.npz", "exactsodshock1d/exactsodshock400.npz")
t, r800, v800, p800 = avg_resid_over_time(directory + "sodshock800.npz", "exactsodshock1d/exactsodshock800.npz")

datasets_higher = [[r200, r400, r800], [v200, v400, v800], [p200, p400, p800]]

def plot_one_set():
    #Plot either the higher- or lower-order sets, but not both
    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(6, 8))
    colors = ['blue', 'red', 'green']
    legend = ['200 grid points', '400 grid points', '800 grid points']

    for i in range(3):
        ax = axs[i]
        for j in range(3):
            ax.plot(t, datasets_lower[i][j], c=colors[j], label=legend[j])
            ax.legend(loc='upper left')
                
            if j == 0: 
                if i == 0: 
                    ax.set_ylabel("Density Residuals")
                if i == 1: ax.set_ylabel("Velocity Residuals")
                if i == 2: ax.set_ylabel("Pressure Residuals")
            if i == 0 and j == 1: ax.set_title("X-Averaged Sod Shock Residuals for Different Grid Sizes")
            if i == 2 and j == 1: ax.set_xlabel("t")

    plt.tight_layout()
    plt.savefig("avgresids_lower.eps", format='eps')
    plt.show()
# plot_one_set()

def plot_both():
    fig, all_axes = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(8, 8))
    colors = ['blue', 'red', 'green']
    legend = ['200 grid points', '400 grid points', '800 grid points']

    for i in range(3):
        axs = all_axes[i]
        for j in range(3):
            axs[0].plot(t, datasets_lower[i][j], c=colors[j], label=legend[j])
            axs[1].plot(t, datasets_higher[i][j], c=colors[j], label=legend[j])
            axs[0].legend(loc='upper left')
            axs[1].legend(loc='upper left')
                
            if j == 0: 
                if i == 0: axs[0].set_ylabel("Density Residuals")
                if i == 1: axs[0].set_ylabel("Velocity Residuals")
                if i == 2: axs[0].set_ylabel("Pressure Residuals")
            if i == 0 and j == 1: 
                axs[0].set_title("Low-Order")
                axs[1].set_title("High-Order")
            if i == 2 and j == 1: 
                axs[0].set_xlabel("t")
                axs[1].set_xlabel("t")

    fig.suptitle("             X-Averaged Absolute Residuals of the Sod Shock Simulation")

    plt.tight_layout()
    plt.savefig("Figures/avgresids_both.png", format='png')
    plt.show()
# plot_both()