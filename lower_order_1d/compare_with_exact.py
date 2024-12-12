"""
Compare simulation results with the exact solution.
"""

from output import *
import numpy as np

#-----200 grid points-----#
# t_e, x_e, rho_e, v_e, p_e = load_data('../exactsodshock1d/exactsodshock200.npz')
# t, x, rho, v, p = load_data("sodshock200.npz")
# # animate(t, x, np.abs(rho - rho_e) / rho_e, np.abs(v - v_e), np.abs(p - p_e) / p_e)

residuals_animation("testsodshock200.npz", '../exactsodshock1d/exactsodshock200.npz', legend1='Simulation', legend2='Exact')

#-----400 grid points-----#
# t_e, x_e, rho_e, v_e, p_e = load_data('../exactsodshock1d/exactsodshock400.npz')
# t, x, rho, v, p = load_data("sodshock400.npz")
# animate(t, x, np.abs(rho - rho_e) / rho_e, np.abs(v - v_e), np.abs(p - p_e) / p_e)

residuals_animation("testsodshock400.npz", '../exactsodshock1d/exactsodshock400.npz', legend1='Simulation', legend2='Exact')

#-----800 grid points-----#
# t_e, x_e, rho_e, v_e, p_e = load_data('../exactsodshock1d/exactsodshock800.npz')
# t, x, rho, v, p = load_data("sodshock800.npz")
# animate(t, x, np.abs(rho - rho_e) / rho_e, np.abs(v - v_e), np.abs(p - p_e) / p_e)

residuals_animation("testsodshock800.npz", '../exactsodshock1d/exactsodshock800.npz', legend1='Simulation', legend2='Exact')