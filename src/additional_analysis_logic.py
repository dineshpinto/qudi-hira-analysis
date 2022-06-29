import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve


class AdditionalAnalysisLogic:
    @staticmethod
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

    @staticmethod
    def antibunching_init(y: np.ndarray) -> list:
        """ Initial guesses for auto-correlation fit. """
        N = 1
        A = 1
        a = np.max(y) - np.min(y)
        tau0 = 0
        tau1 = 20
        tau2 = 30
        return [N, A, a, tau0, tau1, tau2]

    @staticmethod
    def antibunching_function(x: np.ndarray, n: float, a: float, b: float, tau0: float, tau1: float, tau2: float) \
            -> np.ndarray:
        """
        Fit to function
            f(x; n, A, a, tau0, tau1, tau2) =
                a * ((1 - (1+b) * exp(-|x-tau0|/tau1) + a * exp(-|x-tau0|/tau2)) * 1/n + 1 - 1/n)
        """
        return a * ((1 - (1 + b) * np.exp(-abs(x - tau0) / tau1) + b *
                     np.exp(-abs(x - tau0) / tau2)) * 1 / n + 1 - 1 / n)

    def autocorrelation_fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        popt, pcov = curve_fit(self.antibunching_function, x, y, p0=self.antibunching_init(y))
        fit = self.antibunching_function(x, *popt)
        # Anti-bunching at zero delay
        d = {"x": x, "y": y, "fit": fit, "popt": popt, "g2_0": self.antibunching_function(0, *popt)}
        return d

    # Calibration from thermal noise density
    # From Atomic Force Microscopy, Second Edition by Bert Voigtländer
    # Section 11.6.5 Experimental Determination of the Sensitivity and Spring Constant in
    # AFM Without Tip-Sample Contact
    # Eq. 11.28 and 11.26

    @staticmethod
    def power_density(f, N_v_th_exc_square, f_0, Q):
        f_ratio = f / f_0
        return N_v_th_exc_square / ((1 - f_ratio ** 2) ** 2 + 1 / Q ** 2 * f_ratio ** 2)

    @staticmethod
    def s_sensor(N_v_th_exc_square, f_0, T, k, Q):
        k_B = 1.38e-23
        return np.sqrt((2 * k_B * T) / (np.pi * N_v_th_exc_square * k * Q * f_0))

    def find_afm_calibration_parameters(self, data, frequency_range, Q, f_0_guess=44000, T=300, k=5):
        """ Returns dict with calibration (m/V). Also returns fit parameters to power spectral density squared  """
        data = data[(data["Frequency (Hz)"] >= frequency_range[0]) & (data["Frequency (Hz)"] <= frequency_range[1])]
        frequency = data["Frequency (Hz)"].values
        psd_squared = data["Input 1 PowerSpectralDensity (V/sqrt(Hz))"].values ** 2

        popt, pcov = curve_fit(lambda f, n_v_th_exc_square, f0: self.power_density(f, n_v_th_exc_square, f0, Q),
                               xdata=frequency, ydata=psd_squared, p0=[1e-9, f_0_guess])
        N_v_th_exc_square = popt[0]
        f_0 = popt[1]
        calibration = self.s_sensor(N_v_th_exc_square, f_0, T=T, k=k, Q=Q)
        psd_squared_fit = self.power_density(frequency, *popt, Q=Q)

        calibration_params = {"Calibration (m/V)": calibration, "Frequency (Hz)": frequency,
                              "PSD squared (V**2/Hz)": psd_squared, "PSD squared fit (V**2/Hz)": psd_squared_fit}

        return calibration_params
