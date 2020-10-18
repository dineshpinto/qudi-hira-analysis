import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, sparse

import src.io as sio

# Preprocessing techniques from:
# [1] Hopper, D. A., Shulevitz, H. J. & Bassett, L. C. Spin Readout Techniques of the Nitrogen-Vacancy Center in Diamond. Micromachines 9, 437 (2018).
# [2] Paul H. C. Eilers, Hans F.M. Boelens. Baseline Correction with Asymmetric Least Squares Smoothing (2005).

# These global parameters have to be defined when imported
# W  : Integration window width
# dT : Time step
# T0 : Beginning of first integration window
# T1 : Beginning on second integration window
W, dT, T0, T1 = 0, 0, 0, 0

"""
Core functions for preprocessing data.
"""


def find_edge(y, bins=20):
    """ Determine when laser is switched on. """
    h, b = np.histogram(y, bins=bins)
    i0 = int(bins / 2)
    i = h[i0:].argmax() + i0
    threshold = 0.5 * (b[0] + b[i])
    return np.where(y > threshold)[0][0]


def photons_in_window(count_data):
    """ Compute number of photons for |0> and |1> projections. """
    edge = find_edge(count_data.sum(0))

    int_width = W // dT
    int_pos0 = edge + T0 // dT
    int_pos1 = edge + T1 // dT

    if (int_pos1 + int_width) > count_data.shape[1]:
        raise ValueError("Parameters exceed limit.")

    photons_in_window = np.zeros((count_data.shape[0]))
    for idx, photons in enumerate(count_data):
        photons_in_window[idx] = photons[int_pos0:int_pos0 + int_width].sum()
    alpha0, alpha1 = np.array_split(photons_in_window, 2)
    return alpha0, alpha1


def contrast(count_data):
    """ Spin state contrast computation by photon summation (Section II.A, [1]). """
    alpha0, alpha1 = photons_in_window(count_data)
    c = 1 - alpha1 / alpha0
    return c


def shot_noise(count_data):
    """ Photonic shot noise computation using Poisson statistics (square root of counted photons). """
    alpha0, alpha1 = photons_in_window(count_data)
    c = contrast(count_data)
    sn = c * np.sqrt(1 / alpha0 + 1 / alpha1)
    return sn


def signal_to_noise(count_data):
    """ Signal to noise computation (Section II.A, [1]). """
    alpha0, alpha1 = photons_in_window(count_data)
    c = contrast(count_data)
    snr = np.sqrt(alpha0) * c / np.sqrt(2 - c)
    return snr


def bin_data(x, y, num_bins):
    """ Use mean binning technique. """
    if len(x) != len(y):
        raise ValueError("Inputs should be of equal length.")
    if num_bins > len(x):
        raise ValueError("Max bins = ", len(x))
    if num_bins == -1:
        return x, y

    bin_means, bin_edges, bin_number = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2
    return bin_centers, bin_means


def baseline_als(y, lam=1e6, p=0.9, niter=10):
    """ Asymmetric least squares baseline fit [2]. """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def range_bin_and_normalize(x, c, sn, data_range, num_bins, normalize):
    # Select data range
    a, b = data_range[0], data_range[1]
    x, c, sn = x[a:b], c[a:b], sn[a:b]

    # Normalization    
    if normalize:
        base = baseline_als(c)
        c = c / base
        # Under the assumption that d(base)=0
        sn = sn / base

    # Perform binning
    if num_bins != -1:
        _, cb = bin_data(x, c, num_bins)
        xb, snb = bin_data(x, sn, num_bins)
        x, c, sn = xb, cb, snb
    return x, c, sn


"""
Function wrappers for specific measurement sequences.
"""


def time_dependent_measurements(raw_data, dtype=None, data_range=[0, -1], num_bins=-1, normalize=False,
                                old_scheme=False):
    x = copy.deepcopy(raw_data[b"tau"])
    x /= 1e3  # Time in micro seconds
    count_data = raw_data[b"count_data"]
    # Extract data as per data type
    if dtype in ["deer_rabi", "hahn", "t1", "hahn_corr", "t2*", "t1_corr"]:
        if not old_scheme:
            c = contrast(count_data)
            sn = shot_noise(count_data)
        else:
            cc, sn = spin_state(count_data, shot_noise_toggle=True)
            n = len(x)
            c = cc[:n] - cc[n:]
            n1, n2 = sn[:n], sn[n:]
            sn = np.sqrt(n1 ** 2 + n2 ** 2)
    elif dtype in ["rabi"]:
        if old_scheme:
            c, sn = spin_state(count_data, shot_noise_toggle=True)
        else:
            raise NotImplementedError
    elif dtype in ["deer_delay"]:
        s, sn = spin_state(count_data, shot_noise_toggle=True)
        n = len(x)
        yref1, yref2 = s[:n], s[n:2 * n]
        ysig1, ysig2 = s[2 * n:3 * n], s[3 * n:]

        ya = ysig1 - yref1
        yb = ysig2 - yref2
        c = yb - ya

        nref1, nref2 = sn[:n], sn[n:2 * n]
        nsig1, nsig2 = sn[2 * n:3 * n], sn[3 * n:]

        na = np.sqrt(nsig1 ** 2 + nref1 ** 2)
        nb = np.sqrt(nsig2 ** 2 + nref2 ** 2)
        sn = np.sqrt(na ** 2 + nb ** 2)
    else:
        raise KeyError('Invalid dtype, dtype=["deer_rabi", "hahn", "t1", "hahn_corr", "t2*", "t1_corr"]')
    x, c, sn = range_bin_and_normalize(x, c, sn, data_range, num_bins, normalize)
    return x, c, sn


def frequency_dependent_measurements(raw_data, dtype=None, data_range=[0, -1], num_bins=-1, normalize=False):
    x = copy.deepcopy(raw_data[b"frequency"])
    x /= 1e6  # Frequency in MHz
    count_data = raw_data[b"count_data"]
    # Extract data as per data type
    if dtype == "deer_spec":
        s, sn = spin_state(count_data, shot_noise_toggle=True)
    else:
        raise KeyError('Invalid dtype, dtype=["deer_spec"]')
    x, s, sn = range_bin_and_normalize(x, s, sn, data_range, num_bins, normalize)
    return x, s, sn


def raw_counting_measurements(raw_data, dtype=None, data_range=[0, -1], num_bins=-1, normalize=False):
    c = raw_data[b"counts"]
    if dtype == "odmr":
        x = raw_data[b"frequency"]
        x /= 1e6  # Frequency in MHz
        sn = np.sqrt(c)
        x, c, sn = range_bin_and_normalize(x, c, sn, data_range, num_bins, normalize)
    else:
        raise KeyError('Invalid dtype, dtype=["odmr", "autocorrelation"]')
    return x, c, sn


def autocorrelation_measurements(raw_data, dtype=None, data_range=[0, -1], num_bins=-1, normalize=False):
    c = raw_data[b"counts"]
    if dtype == "autocorrelation":
        x = raw_data[b"time_bins"]
        sn = np.sqrt(c)

        a, b = data_range[0], data_range[1]
        x, c, sn = x[a:b], c[a:b], sn[a:b]

        # Normalization    
        if normalize:
            base = baseline_als(c, lam=1e10, p=0.5, niter=10)
            c = c / base
            # Under the assumption that d(base)=0
            sn = sn / base

        # Perform binning
        if num_bins != -1:
            _, cb = bin_data(x, c, num_bins)
            xb, snb = bin_data(x, sn, num_bins)
            x, c, sn = xb, cb, snb
    else:
        raise KeyError('Invalid dtype, dtype=["autocorrelation"]')
    return x, c, sn


"""
The old data analysis method.
"""


# def spin_state(c, shot_noise=False):
#     """
#     Compute the spin state from a 2D array of count data.
#     If AFM is set, we analyze differently and thus return zero (to not trigger the stop_count condition).

#     Parameters
#     ----------
#     c  : count data
#     dT : time step
#     t0 : beginning of integration window relative to the edge
#     t1 : None or beginning of integration window for normalization relative to edge
#     T  : width of integration window

#     Returns
#     -------
#     y       : 1D array that contains the spin state

#     If t1<0, no normalization is performed. If t1>=0, each data point is divided by
#     the value from the second integration window and multiplied with the mean of
#     all normalization windows.
#     """
#     profile = c.sum(0)
#     edge = find_edge(profile)

#     I = int(round(W / dT))
#     i0 = edge + int(round(T0 / dT))
#     y = np.empty((c.shape[0],))

#     for i, slot in enumerate(c):
#         y[i] = slot[i0:i0 + I].sum()

#     if T1 >= 0:
#         i1 = edge + int(round(T1 / float(dT)))
#         y1 = np.empty((c.shape[0],))
#         for i, slot in enumerate(c):
#             y1[i] = slot[i1:i1 + I].sum()
#         if any(y1 * y1.mean() != 0.0):
#             y = y / y1
#         else:
#             raise ValueError("Spin-state computation yielded NaN")
#     else:
#         raise ValueError("Parameter t1 may not be set correctly")

#     num_photons1, num_photons2 = np.zeros_like((c.shape[0])), np.zeros_like((c.shape[0]))
#     if shot_noise:
#         for i, slot in enumerate(c):
#             num_photons1[i] = slot[i0:i0 + I].sum() 
#             num_photons2[i] = slot[i1:i1 + I].sum()
#         noise = y * np.sqrt(1/num_photons1 + 1/num_photons2)
#         return y, noise
#     else:
#         return y

def spin_state(c, shot_noise_toggle=True):
    """
    Compute the spin state and shot noise error from a 2D array of count data.

    Parameters
    ----------
    c  : count data
    dt : time step
    t0 : beginning of integration window relative to the edge
    t1 : beginning of integration window for normalization relative to edge
    T  : width of integration window

    Returns
    -------
    y  : 1D array that contains the spin state
    """
    T = W
    t0 = T0
    t1 = T1
    dt = dT

    profile = c.sum(0)
    edge = find_edge(profile)

    I = int(T / dt)
    i0 = edge + int(t0 / dt)
    i1 = edge + int(t1 / float(dt))

    if (i1 + I) > c.shape[1]:
        raise ValueError("Parameters exceed limit.")

    photons_window1 = np.zeros((c.shape[0]))

    for i, slot in enumerate(c):
        photons_window1[i] = slot[i0:i0 + I].sum()

    if t1 >= 0:
        photons_window2 = np.zeros((c.shape[0]))
        for i, slot in enumerate(c):
            photons_window2[i] = slot[i1:i1 + I].sum()
        if any(photons_window2 * photons_window2.mean() != 0.0):
            state = photons_window1 / photons_window2
        else:
            raise ValueError("Spin-state computation yielded NaN")
    else:
        raise ValueError("Parameter t1 may not be set correctly")

    shot_noise = state * np.sqrt(1 / photons_window1 + 1 / photons_window2)
    if shot_noise_toggle:
        return state, shot_noise
    else:
        return state


def get_all_frq_sweeps(AFM_FOLDER, plot=True):
    files = []

    for file in os.listdir(AFM_FOLDER):
        if file.startswith("frq-sweep") and file.endswith(".dat"):
            files.append(file)

    if plot:
        fig, ax = plt.subplots(nrows=len(files), ncols=2, figsize=(15, len(files) * 3))

    frq_sweep_dict = {}

    for idx, file in enumerate(files):
        params, data = sio.read_dat(AFM_FOLDER + file)
        frq_sweep_dict[file] = {'data': data, 'params': params}

        if plot:
            freq_shift = data["Frequency Shift (Hz)"]
            amplitude = data["Amplitude (m)"]
            phase = data["Phase (deg)"]

            ax[idx, 0].plot(freq_shift, amplitude)
            ax[idx, 0].set_xlabel(data.columns[1])
            ax[idx, 0].set_ylabel(data.columns[2])
            ax[idx, 0].set_title(file)

            ax[idx, 1].plot(freq_shift, phase)
            ax[idx, 1].set_xlabel(data.columns[1])
            ax[idx, 1].set_ylabel(data.columns[3])
            ax[idx, 1].set_title(file)

    return frq_sweep_dict
