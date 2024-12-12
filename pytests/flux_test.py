import numpy as np
from evolution import compute_hll_flux, compute_flux
def test_compute_hll_flux_equal_states():
    gamma = 1.4
    U_L = np.array([[1.0, 0.0, 2.5]])
    U_R = np.array([[1.0, 0.0, 2.5]])  # Same as U_L

    def mock_deconstruct_U(U, gamma=None):
        rho, v, P = 1.0, 0.0, 1.0
        return rho, v, P
    # Mock the deconstruction function
    global deconstruct_U
    deconstruct_U = mock_deconstruct_U

    F_HLL = compute_hll_flux(U_L, U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    assert np.allclose(F_HLL, F_L)  # HLL flux should match flux from identical states

def test_compute_hll_flux_shock():
    gamma = 1.4
    U_L = np.array([[1.0, 1.0, 2.5]])  # Shock wave on the left
    U_R = np.array([[0.1, 0.0, 0.25]])  # Low-pressure region on the right

    def mock_deconstruct_U(U, gamma=None):
        if np.allclose(U, U_L):
            return 1.0, 1.0, 1.0  # Deconstructed left state
        elif np.allclose(U, U_R):
            return 0.1, 0.0, 0.1  # Deconstructed right state
        else:
            raise ValueError("Unexpected state")

    global deconstruct_U
    deconstruct_U = mock_deconstruct_U

    F_HLL = compute_hll_flux(U_L, U_R, gamma)
    assert F_HLL is not None  # Should produce a valid flux without errors
    assert np.all(F_HLL >= 0)  # Ensure positive flux values for shock

def test_compute_hll_flux_small_values():
    gamma = 1.4
    U_L = np.array([[1e-6, 0.0, 1e-6]])  # Small density and pressure
    U_R = np.array([[1e-6, 0.0, 1e-6]])  # Small density and pressure

    def mock_deconstruct_U(U, gamma=None):
        return 1e-6, 0.0, 1e-6

    global deconstruct_U
    deconstruct_U = mock_deconstruct_U

    F_HLL = compute_hll_flux(U_L, U_R, gamma)
    assert F_HLL is not None  # Should handle small values without crashing
    assert np.all(np.isfinite(F_HLL))  # Ensure no NaNs or infinities
