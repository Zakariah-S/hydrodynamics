from output import *
import numpy as np

#-----200 grid points-----#
# t_e, x_e, rho_e, v_e, p_e = load_data('../exactsodshock1d/exactsodshock.npz')
# t, x, rho, v, p = load_data("sodshock.npz")
# # animate(t, x, np.abs(rho - rho_e) / rho_e, np.abs(v - v_e), np.abs(p - p_e) / p_e)

residuals_animation("sodshock.npz", '../exactsodshock1d/exactsodshock.npz', legend1='Simulation', legend2='Exact')

#-----400 grid points-----#
# t_e, x_e, rho_e, v_e, p_e = load_data('../exactsodshock1d/exactsodshockx2.npz')
# t, x, rho, v, p = load_data("sodshockx2.npz")
# animate(t, x, np.abs(rho - rho_e) / rho_e, np.abs(v - v_e), np.abs(p - p_e) / p_e)

residuals_animation("sodshockx2.npz", '../exactsodshock1d/exactsodshockx2.npz', legend1='Simulation', legend2='Exact')

#-----800 grid points-----#
# t_e, x_e, rho_e, v_e, p_e = load_data('../exactsodshock1d/exactsodshockx4.npz')
# t, x, rho, v, p = load_data("sodshockx4.npz")
# animate(t, x, np.abs(rho - rho_e) / rho_e, np.abs(v - v_e), np.abs(p - p_e) / p_e)

residuals_animation("sodshockx4.npz", '../exactsodshock1d/exactsodshockx4.npz', legend1='Simulation', legend2='Exact')