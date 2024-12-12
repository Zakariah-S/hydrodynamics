import numpy as np
import sys
sys.path.insert(1, '..')
from higher_order_1d.evolution import minmod, apply_boundary_conditions, compute_hll_flux

# Unit Test for Minmod Function
def test_minmod():
    assert minmod(1.0, 2.0, 3.0) == 1.0
    assert minmod(-1.0, -2.0, -3.0) == -1.0
    assert minmod(1.0, -2.0, 3.0) == 0.0
    print("Minmod function tests passed.")


def test_minmod_edge_cases():
    assert minmod(0.0, 0.0, 0.0) == 0.0
    assert minmod(2.0, 2.0, 2.0) == 2.0
    assert minmod(-2.0, -2.0, -2.0) == -2.0


    
# # Unit Test for HLL Flux Function
# def test_hll_flux():
#     gamma = 1.4
#     U_L = np.array([1.0, 0.0, 2.5])
#     U_R = np.array([0.125, 0.0, 0.25])
#     F_L = np.array([U_L[1],
#                     U_L[1] ** 2 / U_L[0] + (gamma - 1) * (U_L[2] - 0.5 * U_L[1] ** 2 / U_L[0]),
#                     (U_L[2] + (gamma - 1) * (U_L[2] - 0.5 * U_L[1] ** 2 / U_L[0])) * U_L[1] / U_L[0]])
#     F_R = np.array([U_R[1],
#                     U_R[1] ** 2 / U_R[0] + (gamma - 1) * (U_R[2] - 0.5 * U_R[1] ** 2 / U_R[0]),
#                     (U_R[2] + (gamma - 1) * (U_R[2] - 0.5 * U_R[1] ** 2 / U_R[0])) * U_R[1] / U_R[0]])
#     flux = compute_hll_flux(U_L, U_R, gamma)
#     print("HLL flux computed:", flux)

# Run the tests
# test_minmod()
# test_hll_flux()
