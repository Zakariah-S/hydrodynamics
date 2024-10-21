import numpy as np

# Minmod function
def minmod(a, b, c):
    """
    Minmod limiter function.

    Parameters:
    a, b, c: float
        Input values.

    Returns:
    float
        Minmod result.
    """
    if (a > 0) and (b > 0) and (c > 0):
        return min(a, b, c)
    elif (a < 0) and (b < 0) and (c < 0):
        return max(a, b, c)
    else:
        return 0.0

# Unit Test for Minmod Function
def test_minmod():
    assert minmod(1.0, 2.0, 3.0) == 1.0
    assert minmod(-1.0, -2.0, -3.0) == -1.0
    assert minmod(1.0, -2.0, 3.0) == 0.0
    print("Minmod function tests passed.")

# HLL flux calculation function
def compute_hll_flux(U_L, U_R, F_L, F_R, gamma):
    """
    Compute HLL flux between two states.

    Parameters:
    U_L, U_R: ndarray
        Left and right conserved variables.
    F_L, F_R: ndarray
        Left and right fluxes.
    gamma: float
        Adiabatic index.

    Returns:
    ndarray
        HLL flux.
    """
    # Compute speeds
    rho_L = U_L[0]
    rho_R = U_R[0]
    v_L = U_L[1] / rho_L
    v_R = U_R[1] / rho_R
    P_L = (gamma - 1) * (U_L[2] - 0.5 * rho_L * v_L ** 2)
    P_R = (gamma - 1) * (U_R[2] - 0.5 * rho_R * v_R ** 2)
    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)
    lambda_L = v_L - c_L
    lambda_R = v_R + c_R
    alpha_minus = min(0.0, lambda_L, lambda_R)
    alpha_plus = max(0.0, lambda_L, lambda_R)

    # Compute HLL flux
    flux = (alpha_plus * F_L - alpha_minus * F_R + alpha_plus * alpha_minus * (U_R - U_L)) / (alpha_plus - alpha_minus + 1e-8)
    return flux

# Unit Test for HLL Flux Function
def test_hll_flux():
    gamma = 1.4
    U_L = np.array([1.0, 0.0, 2.5])
    U_R = np.array([0.125, 0.0, 0.25])
    F_L = np.array([U_L[1],
                    U_L[1] ** 2 / U_L[0] + (gamma - 1) * (U_L[2] - 0.5 * U_L[1] ** 2 / U_L[0]),
                    (U_L[2] + (gamma - 1) * (U_L[2] - 0.5 * U_L[1] ** 2 / U_L[0])) * U_L[1] / U_L[0]])
    F_R = np.array([U_R[1],
                    U_R[1] ** 2 / U_R[0] + (gamma - 1) * (U_R[2] - 0.5 * U_R[1] ** 2 / U_R[0]),
                    (U_R[2] + (gamma - 1) * (U_R[2] - 0.5 * U_R[1] ** 2 / U_R[0])) * U_R[1] / U_R[0]])
    flux = compute_hll_flux(U_L, U_R, F_L, F_R, gamma)
    print("HLL flux computed:", flux)

# Run the tests
test_minmod()
test_hll_flux()
