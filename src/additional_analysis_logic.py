from typing import Tuple

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


def _antibunching_init(y: np.ndarray) -> list:
    """ Initial guesses for auto-correlation fit. """
    n = 1
    a = 1
    b = np.max(y) - np.min(y)
    tau0 = 0
    tau1 = 20
    tau2 = 30
    return [n, a, b, tau0, tau1, tau2]


def _antibunching_function(x: np.ndarray, n: float, a: float, b: float, tau0: float, tau1: float, tau2: float) \
        -> np.ndarray:
    """
    Fit to function
        f(x; n, a, tau0, tau1, tau2) =
            a * ((1 - (1+b) * exp(-|x-tau0|/tau1) + a * exp(-|x-tau0|/tau2)) * 1/n + 1 - 1/n)
    """
    return a * ((1 - (1 + b) * np.exp(-np.abs(x - tau0) / tau1) + b *
                 np.exp(-np.abs(x - tau0) / tau2)) * 1 / n + 1 - 1 / n)


def autocorrelation_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    popt, pcov = curve_fit(_antibunching_function, x, y, p0=_antibunching_init(y))
    fit_x, fit_y = x, _antibunching_function(x, *popt)
    result = {"popt": popt, "g2_0": _antibunching_function(0, *popt)}
    return fit_x, fit_y, result


# Calibration from thermal noise density
# From Atomic Force Microscopy, Second Edition by Bert Voigtländer
# Section 11.6.5 Experimental Determination of the Sensitivity and Spring Constant in
# AFM Without Tip-Sample Contact
# Eq. 11.28 and 11.26

def power_density(f, N_v_th_exc_square, f_0, Q):
    f_ratio = f / f_0
    return N_v_th_exc_square / ((1 - f_ratio ** 2) ** 2 + 1 / Q ** 2 * f_ratio ** 2)


def s_sensor(N_v_th_exc_square, f_0, T, k, Q):
    k_B = 1.38e-23
    return np.sqrt((2 * k_B * T) / (np.pi * N_v_th_exc_square * k * Q * f_0))


def find_afm_calibration_parameters(data, frequency_range, Q, f_0_guess=44000, T=300, k=5):
    """ Returns dict with calibration (m/V). Also returns fit parameters to power spectral density squared  """
    data = data[(data["Frequency (Hz)"] >= frequency_range[0]) & (data["Frequency (Hz)"] <= frequency_range[1])]
    frequency = data["Frequency (Hz)"].values
    psd_squared = data["Input 1 PowerSpectralDensity (V/sqrt(Hz))"].values ** 2

    popt, pcov = curve_fit(lambda f, n_v_th_exc_square, f0: power_density(f, n_v_th_exc_square, f0, Q),
                           xdata=frequency, ydata=psd_squared, p0=[1e-9, f_0_guess])
    N_v_th_exc_square = popt[0]
    f_0 = popt[1]
    calibration = s_sensor(N_v_th_exc_square, f_0, T=T, k=k, Q=Q)
    psd_squared_fit = power_density(frequency, *popt, Q=Q)

    calibration_params = {"Calibration (m/V)": calibration, "Frequency (Hz)": frequency,
                          "PSD squared (V**2/Hz)": psd_squared, "PSD squared fit (V**2/Hz)": psd_squared_fit}

    return calibration_params


# from lmfit import Parameters, fit_report, minimize
#
# def autocorrelation(pars, x):
#     """Model a decaying sine wave and subtract data."""
#     vals = pars.valuesdict()
#     n = vals['N']
#     a = vals['A']
#     b = vals['a']
#     tau0 = vals['tau0']
#     tau1 = vals['tau1']
#     tau2 = vals['tau2']
#
#     model =  1 + a * ((1 - (1 + b) * np.exp(-np.abs(x - tau0) / tau1) + b * np.exp(-np.abs(x - tau0) / tau2)) * 1 / n + 1 - 1 / n)
#     return model
#
#
# fit_params = Parameters()
# fit_params.add('N', value=1)
# fit_params.add('A', value=1)
# fit_params.add('a', value=1)
# fit_params.add('tau0', value=1)
# fit_params.add('tau1', value=20)
# fit_params.add('tau2', value=30)
#
# # out = minimize(autocorrelation, fit_params, args=(x,))

