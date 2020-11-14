# -*- coding: utf-8 -*-
"""
The following code is part of qudiamond-analysis under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2020 Dinesh Pinto. See the LICENSE file at the
top-level directory of this distribution and at <https://github.com/dineshpinto/qudiamond-analysis/>
"""

import datetime
import warnings

import matplotlib
import numpy as np
import peakutils
import pytz
import scipy
import scipy.fftpack
from lmfit import Model
from lmfit.models import LinearModel, LorentzianModel, ConstantModel, BreitWignerModel, \
    StepModel, VoigtModel, GaussianModel, ExponentialModel, PowerLawModel, QuadraticModel, PolynomialModel
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .preprocessing import baseline_als

# AFM calibration from thermal noise density:
# Atomic Force Microscopy, Second Edition by Bert Voigtländer
# Section 11.6.5 Experimental Determination of the Sensitivity and Spring Constant in AFM Without Tip-Sample Contact
# Eq. 11.28 and 11.26

"""
Old fitting methods.
"""


def antibunching_init(x, y):
    N = 1
    A = 1
    a = np.max(y) - np.min(y)
    tau0 = 0
    tau1 = 20
    tau2 = 30
    return [N, A, a, tau0, tau1, tau2]


def antibunching(x, N, A, a, tau0, tau1, tau2):
    return A * ((1 - (1 + a) * np.exp(-abs(x - tau0) / tau1) + a * np.exp(-abs(x - tau0) / tau2)) * 1 / N + 1 - 1 / N)


def autocorrelation(x, y):
    popt, pcov = curve_fit(antibunching, x, y, p0=antibunching_init(x, y))
    fit = antibunching(x, *popt)
    d = {"x": x, "y": y, "fit": fit, "popt": popt}
    return d


def peakfinder(x, y, thres=0.9, min_dist=10, plot=True):
    """ Returns indices of peaks in y. Can also plot peaks with markings. """
    indexes = peakutils.indexes(-y, thres=thres / max(y), min_dist=min_dist)
    return indexes


def make_model(num, amplitude, center, width):
    """ Used internally to generate Lorentzian model for each peak. """
    pref = "f{}_".format(num)
    model = LorentzianModel(prefix=pref)
    # model.set_param_hint(pref+'amplitude', value=amplitude[num], min=0, max=2*amplitude[num])
    model.set_param_hint(pref + 'center', value=center[num], min=center[num] - 0.5, max=center[num] + 0.5)
    model.set_param_hint(pref + 'sigma', value=width[num], min=0, max=20)
    return model


def lorentzian_fit(x, y, peaks):
    """ Uses lmfit to fit all peaks to Lorentzian with a constant baseline. """
    num_peaks = len(peaks)
    amplitude = np.zeros(num_peaks) - 1
    width = np.zeros(num_peaks) + 1
    center = x[peaks]

    mod = None
    for i in range(num_peaks):
        this_mod = make_model(i, amplitude, center, width)
        if mod is None:
            mod = this_mod
        else:
            mod = mod + this_mod

    offset = ConstantModel()
    offset.set_param_hint('c', value=1)
    mod = mod + offset

    # Alternate fit methods:
    # 1. Slower but "more" accurate `method='nelder'`
    # 2. Faster but "less" accurate `method='powell'`
    out = mod.fit(y, x=x, method="leastsq")
    return out


def find_peaks2(x, y, num_peaks, thres, min_dist):
    """ Iterates through threshold values to find peaks in data. """
    i = 0
    maxiter = 2000

    while True:
        peak_indexes = peakfinder(x, y, thres=thres, min_dist=min_dist, plot=False)
        if len(peak_indexes) == num_peaks:
            break
        if min_dist > 0:
            if i < maxiter:
                if len(peak_indexes) < num_peaks:
                    thres -= 0.01
                    i += 1
                else:
                    thres += 0.01
                    i += 1
            else:
                min_dist -= 1
                i = 0
        else:
            raise ValueError("Peaks could not be determined."
                             f"Maximum iterations ({maxiter}) exceeded. thres={thres}, min_dist={min_dist}")
    return peak_indexes


def spectroscopy(x, y, n, num_peaks, num_bins=-1, arr_range=[0, -1], dtype=None, height=-0.9, width=2):
    """
    Analyzes spectroscopic data, performs the following operations:
        1. Extracts (x, y) from binary data
        2. Finds y baseline and subtracts it
        3. Finds peaks in data iteratively
        4. Generates a Lorentzian fitting model
        5. Fits data to model
        6. Calculates standard error and signal to noise ratio
    Returns a dictionary.
    """
    bigdict = {"x": x, "y": y, "n": n}

    if num_peaks == 0:
        return bigdict

    peak_indexes, _ = find_peaks(-y, height=height, width=width)

    if len(peak_indexes) != num_peaks:
        if len(peak_indexes) > num_peaks:
            error_message = "Try decreasing height or increasing width."
        else:
            error_message = "Try increasing height or decreasing width."
        raise ValueError("Incorrect number of peaks ({}). ".format(len(peak_indexes)) + error_message)

    opt = lorentzian_fit(x, y, peak_indexes)
    fit = opt.best_fit
    fit_report = opt.fit_report()

    x_int = np.linspace(x.min(), x.max(), len(x) * 8)
    fit_interpolant = interp1d(x, fit, kind='cubic')
    fit_int = fit_interpolant(x_int)

    bigdict["fit_int"] = fit_int
    bigdict["x_int"] = x_int
    bigdict["peak_indexes"] = peak_indexes
    bigdict["peaks"] = x[peak_indexes]
    bigdict["fit"] = fit
    bigdict["fit_report"] = fit_report

    return bigdict


def exp(x, c, A, d, a=1):
    return c + A * np.exp(-(x / d) ** a)


def exphahn_init(x, y, decay, period):
    c = np.mean(y)
    A = np.std(y) * np.sqrt(2)
    d = decay

    t = period
    # f = 1 / p
    init = [c, A, d, t]
    return init


def exp_init(x, y, decay):
    c = np.mean(y)
    A = np.std(y) * np.sqrt(2)
    d = decay
    init = [c, A, d]
    return init


def exphahn(x, c, A, d, t):
    return c + A * np.exp(-(x / d)) * np.cos(2 * np.pi / t * x)


def hahn_decay(x, y, n, dtype=None, decay=10, period=15, contrast_shift=1, num_bins=-1, arr_range=[0, -1],
               revivals=False, base=False):
    bigdict = {}

    bigdict["x"] = x
    bigdict["y"] = y

    if base:
        baseline = baseline_als(y, lam=1e6, p=0.01)
        y = y - baseline

    if revivals:
        init = exphahn_init(x, y, decay=decay, period=period)
        popt, pcov = curve_fit(exphahn, x, y, p0=init)
        fit = exphahn(x, *popt)
        fit_exp = exp(x, *popt[0:3])
    else:
        init = exp_init(x, y, decay=decay)
        popt, pcov = curve_fit(exp, x, y, p0=init)
        fit = exp(x, *popt)
        fit_exp = fit

    y_contrast = y * 100 + contrast_shift
    n_contrast = n * 100
    fit_exp_contrast = fit_exp * 100 + contrast_shift
    fit_contrast = fit * 100 + contrast_shift

    x_int = np.linspace(x.min(), x.max(), len(x) * 4)

    fit_interpolant = interp1d(x, fit, kind='cubic')
    fit_int = fit_interpolant(x_int)

    fit_contrast_interpolant = interp1d(x, fit_contrast, kind='cubic')
    fit_contrast_int = fit_contrast_interpolant(x_int)

    fit_exp_contrast_interpolant = interp1d(x, fit_exp_contrast, kind='cubic')
    fit_exp_contrast_int = fit_exp_contrast_interpolant(x_int)

    bigdict["y_contrast"] = y_contrast
    bigdict["fit"] = fit
    bigdict["fit_exp"] = fit_exp
    bigdict["fit_exp_contrast"] = fit_exp_contrast
    bigdict["fit_exp_contrast_int"] = fit_exp_contrast_int
    bigdict["fit_contrast"] = fit_contrast
    bigdict["n"] = n
    bigdict["x_int"] = x_int
    bigdict["fit_contrast_int"] = fit_contrast_int
    bigdict["fit_int"] = fit_int
    bigdict["decay"] = init[2]
    bigdict["n_contrast"] = n_contrast
    bigdict["popt"] = popt
    bigdict["perr"] = np.sqrt(np.diag(pcov))

    return bigdict


def expsine(x, c, A, d, w, p):
    return c + A * np.exp(- x / d) * np.sin(2 * np.pi * x / w + p)


def expsine_init(x, y, decay):
    c = np.mean(y)
    A = np.std(y) * np.sqrt(2)
    freq = np.fft.rfftfreq(len(x), (x[1] - x[0]))
    w = abs(freq[np.argmax(abs(np.fft.rfft(x))[1:]) + 1])
    init = [c, A, decay, 1 / w, np.pi]
    return init


def take_fft(x, y):
    N = len(x)
    dx = x[1] - x[0]
    yf = scipy.fftpack.fft(y)
    y_fft = 2 / N * np.abs(yf[:N // 2])[1:]
    x_fft = np.linspace(0, 1 / (2 * dx), N // 2)[1:]
    return x_fft, y_fft


def rabi_oscillations(x, y, n, dtype="rabi", decay=5, contrast_shift=1, num_bins=-1, arr_range=[0, -1], base=False):
    bigdict = {}

    if base:
        # Use with caution. Rabi should not require a baseline correction.
        # The parameters are set to avoid overcorrecting.
        base2 = baseline_als(y, lam=1e6, p=0.5)
        y = y - base2

    bigdict["x"] = x
    bigdict["y"] = y
    bigdict["n"] = n

    init = expsine_init(x, y, decay)

    popt, pcov = curve_fit(expsine, x, y, p0=init)

    fit = expsine(x, *popt)
    # exp1, exp2 = exp(x, *popt)
    y_contrast = y * 100 + contrast_shift
    n_contrast = n * 100
    fit_contrast = fit * 100 + contrast_shift

    fit_contrast_interpolant = interp1d(x, fit_contrast, kind='cubic')
    x_int = np.linspace(x.min(), x.max(), 1000)
    fit_contrast_int = fit_contrast_interpolant(x_int)

    fit_interpolant = interp1d(x, fit, kind='cubic')
    fit_int = fit_interpolant(x_int)

    x_fft, y_fft = take_fft(x, y_contrast)

    bigdict["x_fft"] = x_fft
    bigdict["y_fft"] = y_fft
    bigdict["y_contrast"] = y_contrast
    bigdict["fit"] = fit
    bigdict["fit_contrast"] = fit_contrast
    bigdict["n"] = n
    bigdict["x_int"] = x_int
    bigdict["fit_int"] = fit_int
    bigdict["fit_contrast_int"] = fit_contrast_int
    bigdict["decay"] = init[2]
    bigdict["n_contrast"] = n_contrast
    # bigdict["exp1"] = exp1
    # bigdict["exp2"] = exp2
    bigdict["popt"] = popt
    bigdict["perr"] = np.sqrt(np.diag(pcov))

    return bigdict


# Fitting AFM frequency sweep data


def offset_type(linear_offset):
    if linear_offset:
        return LinearModel()
    else:
        return ConstantModel()


def fit_fano(x, y, linear_offset=False):
    fano = BreitWignerModel()
    params = fano.guess(y, x=x)

    offset = offset_type(linear_offset)
    params += offset.guess(y, x=x)

    model = fano + offset
    out = model.fit(y, x=x, params=params, method="leastsq")
    return out


def fit_lorentzian(x, y, linear_offset=False):
    lorentzian = LorentzianModel()
    params = lorentzian.guess(y, x=x)

    offset = offset_type(linear_offset)
    params += offset.guess(y, x=x)

    model = lorentzian + offset
    out = model.fit(y, x=x, params=params, method="leastsq")
    return out


def fit_gaussian(x, y, linear_offset=False):
    gaussian = GaussianModel()
    params = gaussian.guess(-y, x=x)

    offset = offset_type(linear_offset)
    params += offset.guess(y, x=x)

    model = gaussian + offset
    out = model.fit(y, x=x, method="leastsq")
    return out


def fit_voigt(x, y, linear_offset=False):
    voigt = VoigtModel()
    params = voigt.guess(-y, x=x)

    offset = offset_type(linear_offset)
    params += offset.guess(y, x=x)

    model = voigt + offset
    out = model.fit(y, x=x, params=params, method="leastsq")
    return out


def fit_logistic(x, y, linear_offset=False):
    step = StepModel(form='logistic')
    params = step.guess(y, x=x)

    offset = offset_type(linear_offset)
    params += offset.guess(y, x=x)

    model = step + offset
    out = model.fit(y, x=x, params=params, method="leastsq")
    return out


# Calibration from thermal noise density
# From Atomic Force Microscopy, Second Edition by Bert Voigtländer
# Section 11.6.5 Experimental Determination of the Sensitivity and Spring Constant in AFM Without Tip-Sample Contact
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


def func_linear(x, a, b):
    """ Simple linear function to use for scipy.curve_fit. """
    return a + b * x


def func_exponential(x, a, b, c):
    """ Simple exponential function to use for scipy.curve_fit. """
    return a + b * np.exp(c * x)


def func_negexponential(x, a, b, c):
    """ Simple exponential function to use for scipy.curve_fit. """
    return a + b * np.exp(-c * x)


def func_logarithmic(x, a, b, c):
    """ Simple logarithmic function to use for scipy.curve_fit. """
    return a + b * np.log(c * x)


def func_neglogarithmic(x, a, b, c):
    """ Simple logarithmic function to use for scipy.curve_fit. """
    return a + b * np.log(-c * x)


def func_offset_exponentional(x, a, b, c, offset):
    return a + b * np.exp(-c * (x - offset))


def time_extrapolation(df, ylabel, end_date=None, start_index=0, fit="linear"):
    """
    DEPRECATED in favor of time_extrapolation_lmfit
    Function to perform a extrapolation in time on a DataFrame.
    Function choices for extrapolation: linear, exponential, logarithmic or custom function (set fit to function)
    """
    warnings.warn("time_extrapolation() is deprecated; use time_extrapolation_lmfit().", DeprecationWarning)
    # Choose a starting point for the fitting
    dfc = df[start_index:]
    # Convert matplotlib dates to datetime objects, returns a tz-aware object
    start_extrap = matplotlib.dates.num2date(dfc["MPL_datetimes"].values[0], tz=pytz.timezone('Europe/Berlin'))
    # Add tz-awareness information to datetime object to prevent tz-naive and tz-aware conflicts
    end_extrap = datetime.datetime.strptime(end_date, "%d-%b-%y %H:%M").replace(tzinfo=pytz.timezone('Europe/Berlin'))

    # Get matplotlib date series
    duration_in_sec = (end_extrap - start_extrap).total_seconds()
    duration_in_h = int(divmod(duration_in_sec, 3600)[0])

    date_generated = [start_extrap + datetime.timedelta(hours=x) for x in range(0, duration_in_h)]
    dt_series_mpl = matplotlib.dates.date2num(date_generated)

    # Fit date series with a choice of functions
    if fit == "linear":
        popt, pcov = curve_fit(func_linear, xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func_linear(dt_series_mpl, *popt)
    elif fit == "exponentional":
        popt, pcov = curve_fit(func_exponential, xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func_exponential(dt_series_mpl, *popt)
    elif fit == "negexponentional":
        popt, pcov = curve_fit(func_negexponential, xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func_negexponential(dt_series_mpl, *popt)
    elif fit == "logarithmic":
        popt, pcov = curve_fit(func_logarithmic, xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func_logarithmic(dt_series_mpl, *popt)
    elif fit == "neglogarithmic":
        popt, pcov = curve_fit(func_neglogarithmic, xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func_neglogarithmic(dt_series_mpl, *popt)
    elif fit == "offset_exponentional":
        popt, pcov = curve_fit(lambda x, a, b, c: func_offset_exponentional(x, a, b, c, offset=dfc["MPL_datetimes"][0]),
                               xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func_offset_exponentional(dt_series_mpl, *popt, offset=dfc["MPL_datetimes"][0])
    elif hasattr(fit, '__call__'):
        # Use a custom function for fitting
        func = fit
        popt, pcov = curve_fit(func, xdata=dfc["MPL_datetimes"], ydata=dfc[ylabel])
        fit_result = func(dt_series_mpl, *popt)
    else:
        raise NotImplementedError("Fitting method '{}' not implemented".format(fit))
    return dt_series_mpl, fit_result


def time_extrapolation_lmfit(df, ylabel, end_date=None, start_index=0, fit="linear"):
    """
    Extrapolate a set of time series data.

    Parameters
    ----------
    df : pandas:DataFrame
        containing column 'MPL_datetimes'
    ylabel : string
        Label in DataFame corresponding to y-axis
    end_date : string, dd-mmm-yy H:M
        Date to extrapolate to
    start_index : int
        DataFrame index to start fitting from
    fit : function or {"linear", "quadratic", "polynomial<n>" (n=degree), "powerlaw", "exponentional"}

    Returns
    -------
    extrapolated_dates_mpl : numpy.ndarray
        extrapolated matplotlib dates

    extrapolation : numpy.ndarray
        extrapolated y-data

    result : lmfit.Model
        use result.best_fit to get optimal fit data
    """

    # Choose a starting point for the fitting
    dfc = df[start_index:]
    # Convert matplotlib dates to datetime objects, returns a tz-aware object
    start_extrap = matplotlib.dates.num2date(dfc["MPL_datetimes"].values[0], tz=pytz.timezone('Europe/Berlin'))
    # Add tz-awareness information to datetime object to prevent tz-naive and tz-aware conflicts
    end_extrap = datetime.datetime.strptime(end_date, "%d-%b-%y %H:%M").replace(tzinfo=pytz.timezone('Europe/Berlin'))

    # Get matplotlib date series
    duration_in_sec = (end_extrap - start_extrap).total_seconds()
    duration_in_h = int(divmod(duration_in_sec, 3600)[0])

    extrapolated_dates_datetime = [start_extrap + datetime.timedelta(hours=x) for x in range(0, duration_in_h)]
    extrapolated_dates_mpl = matplotlib.dates.date2num(extrapolated_dates_datetime)

    # Fit date series with a choice of lmfit functions
    if hasattr(fit, '__call__'):
        # Use a custom function for fitting
        mod = Model(fit)
        result = mod.fit(dfc[ylabel], x=dfc["MPL_datetimes"])
    elif fit == "linear":
        mod = LinearModel()
        pars = mod.guess(dfc[ylabel], x=dfc["MPL_datetimes"])
        result = mod.fit(dfc[ylabel], pars, x=dfc["MPL_datetimes"])
    elif fit == "quadratic":
        mod = QuadraticModel()
        pars = mod.guess(dfc[ylabel], x=dfc["MPL_datetimes"])
        result = mod.fit(dfc[ylabel], pars, x=dfc["MPL_datetimes"])
    elif fit.startswith("polynomial"):
        degree = int(fit[-1])
        mod = PolynomialModel(degree=degree)
        pars = mod.guess(dfc[ylabel], x=dfc["MPL_datetimes"])
        result = mod.fit(dfc[ylabel], pars, x=dfc["MPL_datetimes"])
    elif fit == "powerlaw":
        mod = PowerLawModel()
        pars = mod.guess(dfc[ylabel], x=dfc["MPL_datetimes"])
        result = mod.fit(dfc[ylabel], pars, x=dfc["MPL_datetimes"])
    elif fit == "exponentional":
        mod = ExponentialModel()
        pars = mod.guess(dfc[ylabel], x=dfc["MPL_datetimes"])
        result = mod.fit(dfc[ylabel], pars, x=dfc["MPL_datetimes"])
    else:
        raise NotImplementedError("Fitting method '{}' not implemented, use a custom "
                                  "function parameter fit=<func> instead".format(fit))
    # Extrapolate date from best fit parameters
    extrapolation = mod.eval(params=result.params, x=extrapolated_dates_mpl)

    return extrapolated_dates_mpl, extrapolation, dfc["MPL_datetimes"], result


def find_nearest(array, value):
    """
    Find nearest element to a value in a given array.

    Args:
        array: np.ndarray, list
            array to find element in
        value: int, float
            value to match element with

    Returns:
        idx: int
            index of array element
        array[idx]: int, float
            value of array element

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def setpointy_reach_time(x, y, setpointy):
    """
    Find time to reach a given y-setpoint. Useful for future predictions on extrapolated data.

    Args:
        x: np.ndarray, list
            Time axis as matplotlib datetimes
        y: np.ndarray, list
            Independent data
        setpointy: int, float
            setpoint value to find (exactly or nearest) in y data

    Returns:
        closest_val_dt: datetime.DateTime
            value of the time axis corresponding to the y setpoint

    """
    closest_val_idx, closest_val = find_nearest(y, setpointy)
    closest_val_dt = matplotlib.dates.num2date(x[closest_val_idx])
    return closest_val_dt


