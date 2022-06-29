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

Copyright (c) 2022 Dinesh Pinto. See the LICENSE file at the
top-level directory of this distribution and at
<https://github.com/dineshpinto/qudiamond-analysis/>
"""

from typing import Tuple

import numpy as np
import pandas as pd
from lmfit.model import ModelResult

from src.fit_logic import FitLogic


def perform_fit(
        x: pd.Series,
        y: pd.Series,
        fit_function: str,
        estimator: str = "generic",
        dims: str = "1d") -> Tuple[np.ndarray, np.ndarray, ModelResult]:
    """
    Args:
        x: x-data
        y: y-data
        fit_function:
            - decayexponential
            - decayexponentialstretched
            - sineexponentialdecay
            - sinedouble
            - sinedoublewithexpdecay
            - sinedoublewithtwoexpdecay
        estimator:
            - generic
            - dip
        dims:
            - 1d
            - 2d
    Returns:
        fit_x, fit_y, result
    """

    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    f = FitLogic()
    fit = {dims: {'default': {'fit_function': fit_function, 'estimator': estimator}}}
    user_fit = f.validate_load_fits(fit)

    use_settings = {}
    for key in user_fit[dims]["default"]["parameters"].keys():
        use_settings[key] = False
    user_fit[dims]["default"]["use_settings"] = use_settings

    fc = f.make_fit_container("test", dims)
    fc.set_fit_functions(user_fit[dims])
    fc.set_current_fit("default")
    fc.use_settings = None
    fit_x, fit_y, result = fc.do_fit(x, y)
    return fit_x, fit_y, result


def get_fits(dim: str = "1d") -> list:
    return FitLogic().fit_list[dim].keys()


def analyse_mean(
        laser_data: np.ndarray,
        signal_start: float = 0.0,
        signal_end: float = 200e-9,
        bin_width: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    # Get number of lasers
    num_of_lasers = laser_data.shape[0]

    if not isinstance(bin_width, float):
        return np.zeros(num_of_lasers), np.zeros(num_of_lasers)

    # Convert the times in seconds to bins (i.e. array indices)
    signal_start_bin = round(signal_start / bin_width)
    signal_end_bin = round(signal_end / bin_width)

    # initialize data arrays for signal and measurement error
    signal_data = np.empty(num_of_lasers, dtype=float)
    error_data = np.empty(num_of_lasers, dtype=float)

    # loop over all laser pulses and analyze them
    for ii, laser_arr in enumerate(laser_data):
        # calculate the mean of the data in the signal window
        signal = laser_arr[signal_start_bin:signal_end_bin].mean()
        signal_sum = laser_arr[signal_start_bin:signal_end_bin].sum()
        signal_error = np.sqrt(signal_sum) / (signal_end_bin - signal_start_bin)

        # Avoid numpy C type variables overflow and NaN values
        if signal < 0 or signal != signal:
            signal_data[ii] = 0.0
            error_data[ii] = 0.0
        else:
            signal_data[ii] = signal
            error_data[ii] = signal_error

    return signal_data, error_data


def analyse_mean_reference(
        laser_data: np.ndarray,
        signal_start: float = 0.0,
        signal_end: float = 200e-9,
        norm_start: float = 300e-9,
        norm_end: float = 500e-9,
        bin_width: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method takes the mean of the signal window.
    It then does not divide by the background window to normalize
    but rather substracts the background window to generate the output.
    """
    # Get number of lasers
    num_of_lasers = laser_data.shape[0]

    if not isinstance(bin_width, float):
        return np.zeros(num_of_lasers), np.zeros(num_of_lasers)

    # Convert the times in seconds to bins (i.e. array indices)
    signal_start_bin = round(signal_start / bin_width)
    signal_end_bin = round(signal_end / bin_width)
    norm_start_bin = round(norm_start / bin_width)
    norm_end_bin = round(norm_end / bin_width)

    # initialize data arrays for signal and measurement error
    signal_data = np.empty(num_of_lasers, dtype=float)
    error_data = np.empty(num_of_lasers, dtype=float)

    # loop over all laser pulses and analyze them
    for ii, laser_arr in enumerate(laser_data):
        # calculate the sum and mean of the data in the normalization window
        tmp_data = laser_arr[norm_start_bin:norm_end_bin]
        reference_sum = np.sum(tmp_data)
        reference_mean = (reference_sum / len(tmp_data)) if len(tmp_data) != 0 else 0.0

        # calculate the sum and mean of the data in the signal window
        tmp_data = laser_arr[signal_start_bin:signal_end_bin]
        signal_sum = np.sum(tmp_data)
        signal_mean = (signal_sum / len(tmp_data)) if len(tmp_data) != 0 else 0.0

        signal_data[ii] = signal_mean - reference_mean

        # calculate with respect to gaussian error 'evolution'
        error_data[ii] = signal_data[ii] * np.sqrt(1 / abs(signal_sum) + 1 / abs(reference_sum))

    return signal_data, error_data


def analyse_mean_norm(
        laser_data: np.ndarray,
        signal_start: float = 0.0,
        signal_end: float = 200e-9,
        norm_start: float = 300e-9,
        norm_end=500e-9,
        bin_width: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    # Get number of lasers
    num_of_lasers = laser_data.shape[0]

    if not isinstance(bin_width, float):
        return np.zeros(num_of_lasers), np.zeros(num_of_lasers)

    # Convert the times in seconds to bins (i.e. array indices)
    signal_start_bin = round(signal_start / bin_width)
    signal_end_bin = round(signal_end / bin_width)
    norm_start_bin = round(norm_start / bin_width)
    norm_end_bin = round(norm_end / bin_width)

    # initialize data arrays for signal and measurement error
    signal_data = np.empty(num_of_lasers, dtype=float)
    error_data = np.empty(num_of_lasers, dtype=float)

    # loop over all laser pulses and analyze them
    for ii, laser_arr in enumerate(laser_data):
        # calculate the sum and mean of the data in the normalization window
        tmp_data = laser_arr[norm_start_bin:norm_end_bin]
        reference_sum = np.sum(tmp_data)
        reference_mean = (reference_sum / len(tmp_data)) if len(tmp_data) != 0 else 0.0

        # calculate the sum and mean of the data in the signal window
        tmp_data = laser_arr[signal_start_bin:signal_end_bin]
        signal_sum = np.sum(tmp_data)
        signal_mean = (signal_sum / len(tmp_data)) if len(tmp_data) != 0 else 0.0

        # Calculate normalized signal while avoiding division by zero
        if reference_mean > 0 and signal_mean >= 0:
            signal_data[ii] = signal_mean / reference_mean
        else:
            signal_data[ii] = 0.0

        # Calculate measurement error while avoiding division by zero
        if reference_sum > 0 and signal_sum > 0:
            # calculate with respect to gaussian error 'evolution'
            error_data[ii] = signal_data[ii] * np.sqrt(1 / signal_sum + 1 / reference_sum)
        else:
            error_data[ii] = 0.0

    return signal_data, error_data
