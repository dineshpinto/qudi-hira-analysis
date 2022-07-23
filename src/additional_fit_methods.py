import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve


def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.9, niter: int = 10):
    """ Asymmetric least squares baseline fit. """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    z = None
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


# Calibration from thermal noise density
# From Atomic Force Microscopy, Second Edition by Bert Voigtländer
# Section 11.6.5 Experimental Determination of the Sensitivity and Spring Constant in
# AFM Without Tip-Sample Contact
# Eq. 11.28 and 11.26


def power_density(f, N_v_th_exc_square, resonance_freq, q_factor):
    f_ratio = f / resonance_freq
    return N_v_th_exc_square / ((1 - f_ratio ** 2) ** 2 + 1 / q_factor ** 2 * f_ratio ** 2)


def s_sensor(N_v_th_exc_square, resonance_freq, temperature, spring_constant, q_factor):
    boltzmann_constant = 1.38e-23
    return np.sqrt((2 * boltzmann_constant * temperature) / (
                np.pi * N_v_th_exc_square * spring_constant * q_factor * resonance_freq))


def find_afm_calibration_parameters(data, frequency_range, q_factor, resonance_freq_guess=44000, temperature=300,
                                    spring_constant=5):
    """ Returns dict with calibration (m/V). Also returns fit parameters to power spectral density squared  """
    data = data[(data["Frequency (Hz)"] >= frequency_range[0]) & (data["Frequency (Hz)"] <= frequency_range[1])]
    frequency = data["Frequency (Hz)"].values
    psd_squared = data["Input 1 PowerSpectralDensity (V/sqrt(Hz))"].values ** 2

    popt, pcov = curve_fit(lambda f, n_v_th_exc_square, f0: power_density(f, n_v_th_exc_square, f0, q_factor),
                           xdata=frequency, ydata=psd_squared, p0=[1e-9, resonance_freq_guess])
    N_v_th_exc_square = popt[0]
    f_0 = popt[1]
    calibration = s_sensor(N_v_th_exc_square, f_0, temperature=temperature, spring_constant=spring_constant,
                           q_factor=q_factor)
    psd_squared_fit = power_density(frequency, *popt, q_factor=q_factor)

    calibration_params = {"Calibration (m/V)": calibration, "Frequency (Hz)": frequency,
                          "PSD squared (V**2/Hz)": psd_squared, "PSD squared fit (V**2/Hz)": psd_squared_fit}

    return calibration_params
