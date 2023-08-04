from __future__ import annotations

import logging
import random
import re
from itertools import product
from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

import qudi_hira_analysis.raster_odmr_fitting as rof
from qudi_hira_analysis.qudi_fit_logic import FitLogic

if TYPE_CHECKING:
    from lmfit.model import ModelResult, Parameter
    from .measurement_dataclass import MeasurementDataclass

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class FitMethodsAndEstimators:
    # Fit methods with corresponding estimators
    antibunching: tuple = ("antibunching", "dip")
    hyperbolicsaturation: tuple = ("hyperbolicsaturation", "generic")
    lorentzian: tuple = ("lorentzian", "dip")
    lorentziandouble: tuple = ("lorentziandouble", "dip")
    sineexponentialdecay: tuple = ("sineexponentialdecay", "generic")
    decayexponential: tuple = ("decayexponential", "generic")
    gaussian: tuple = ("gaussian", "dip")
    gaussiandouble: tuple = ("gaussiandouble", "dip")
    gaussianlinearoffset: tuple = ("gaussianlinearoffset", "dip")
    lorentziantriple: tuple = ("lorentziantriple", "dip")
    biexponential: tuple = ("biexponential", "generic")
    decayexponentialstretched: tuple = ("decayexponentialstretched", "generic")
    linear: tuple = ("linear", "generic")
    sine: tuple = ("sine", "generic")
    sinedouble: tuple = ("sinedouble", "generic")
    sinedoublewithexpdecay: tuple = ("sinedoublewithexpdecay", "generic")
    sinedoublewithtwoexpdecay: tuple = ("sinedoublewithtwoexpdecay", "generic")
    sinestretchedexponentialdecay: tuple = ("sinestretchedexponentialdecay", "generic")
    sinetriple: tuple = ("sinetriple", "generic")
    sinetriplewithexpdecay: tuple = ("sinetriplewithexpdecay", "generic")
    sinetriplewiththreeexpdecay: tuple = ("sinetriplewiththreeexpdecay", "generic")
    twoDgaussian: tuple = ("twoDgaussian", "generic")


class AnalysisLogic(FitLogic):
    fit_function = FitMethodsAndEstimators

    def __init__(self):
        super().__init__()
        self.log = logging.getLogger(__name__)

    def _perform_fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            fit_function: str,
            estimator: str,
            parameters: list[Parameter] = None,
            dims: str = "1d") -> Tuple[np.ndarray, np.ndarray, ModelResult]:
        """
        Fits available:
            | Dimension | Fit                           |
            |-----------|-------------------------------|
            | 1d        | decayexponential              |
            |           | biexponential                 |
            |           | decayexponentialstretched     |
            |           | gaussian                      |
            |           | gaussiandouble                |
            |           | gaussianlinearoffset          |
            |           | hyperbolicsaturation          |
            |           | linear                        |
            |           | lorentzian                    |
            |           | lorentziandouble              |
            |           | lorentziantriple              |
            |           | sine                          |
            |           | sinedouble                    |
            |           | sinedoublewithexpdecay        |
            |           | sinedoublewithtwoexpdecay     |
            |           | sineexponentialdecay          |
            |           | sinestretchedexponentialdecay |
            |           | sinetriple                    |
            |           | sinetriplewithexpdecay        |
            |           | sinetriplewiththreeexpdecay   |
            | 2d        | twoDgaussian                  |
        Estimators:
            - generic
            - dip
        """
        fit = {dims: {'default': {'fit_function': fit_function, 'estimator': estimator}}}
        user_fit = self.validate_load_fits(fit)

        if parameters:
            user_fit[dims]["default"]["parameters"].add_many(*parameters)

        use_settings = {}
        for key in user_fit[dims]["default"]["parameters"].keys():
            if parameters:
                if key in [p.name for p in parameters]:
                    use_settings[key] = True
                else:
                    use_settings[key] = False
            else:
                use_settings[key] = False
        user_fit[dims]["default"]["use_settings"] = use_settings

        fc = self.make_fit_container("test", dims)
        fc.set_fit_functions(user_fit[dims])
        fc.set_current_fit("default")
        fit_x, fit_y, result = fc.do_fit(x, y)
        return fit_x, fit_y, result

    def fit(
            self,
            x: str | np.ndarray | pd.Series,
            y: str | np.ndarray | pd.Series,
            fit_function: FitMethodsAndEstimators,
            data: pd.DataFrame = None,
            parameters: list[Parameter] = None
    ) -> Tuple[np.ndarray, np.ndarray, ModelResult]:
        if "twoD" in fit_function[0]:
            dims = "2d"
        else:
            dims = "1d"

        if data is None:
            if isinstance(x, pd.Series) or isinstance(x, pd.Index):
                x = x.to_numpy()
            if isinstance(y, pd.Series):
                y = y.to_numpy()
        elif isinstance(data, pd.DataFrame):
            x = data[x].to_numpy()
            y = data[y].to_numpy()
        else:
            raise TypeError("Data must be a pandas DataFrame or None")

        return self._perform_fit(
            x=x,
            y=y,
            fit_function=fit_function[0],
            estimator=fit_function[1],
            parameters=parameters,
            dims=dims
        )

    def get_all_fits(self) -> Tuple[list, list]:
        one_d_fits = list(self.fit_list['1d'].keys())
        two_d_fits = list(self.fit_list['2d'].keys())
        self.log.info(f"1d fits: {one_d_fits}\n2d fits: {two_d_fits}")
        return one_d_fits, two_d_fits

    @staticmethod
    def analyse_mean(
            laser_data: np.ndarray,
            signal_start: float = 100e-9,
            signal_end: float = 300e-9,
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

    @staticmethod
    def analyse_mean_reference(
            laser_data: np.ndarray,
            signal_start: float = 100e-9,
            signal_end: float = 300e-9,
            norm_start: float = 1000e-9,
            norm_end: float = 2000e-9,
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

    @staticmethod
    def analyse_mean_norm(
            laser_data: np.ndarray,
            signal_start: float = 100e-9,
            signal_end: float = 300e-9,
            norm_start: float = 1000e-9,
            norm_end=2000e-9,
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

    def optimize_hyperparameters(
            self,
            measurements: dict[str, MeasurementDataclass],
            num_samples: int = 10,
            num_params: int = 3,
    ) -> Tuple[float, Tuple[float, float, float]]:
        """
        This method optimizes the hyperparameters of the ODMR analysis.
        It does so by randomly sampling a subset of the measurements and
        then optimizing the hyperparameters for them.

        Args:
            measurements: A dictionary of measurements to optimize the hyperparameters for.
            num_params: The number of parameters to optimize.
            num_samples: The number of measurements to sample.

        Returns:
            The optimal hyperparameters.
        """
        r2_threshs = np.around(np.linspace(start=0.9, stop=0.99, num=num_params), decimals=2)
        thresh_fracs = np.around(np.linspace(start=0.5, stop=0.9, num=num_params), decimals=1)
        sigma_thresh_fracs = np.around(np.linspace(start=0.1, stop=0.2, num=num_params), decimals=1)

        odmr_sample = {}
        for k, v in random.sample(sorted(measurements.items()), k=num_samples):
            odmr_sample[k] = v

        highest_min_r2 = 0
        optimal_params = (0, 0, 0)

        for idx, (r2_thresh, thresh_frac, sigma_thresh_frac) in enumerate(
                product(r2_threshs, thresh_fracs, sigma_thresh_fracs)):
            odmr_sample = self.raster_odmr_fitting(
                odmr_sample,
                r2_thresh=r2_thresh,
                thresh_frac=thresh_frac,
                sigma_thresh_frac=sigma_thresh_frac,
                min_thresh=0.01,
                progress_bar=False
            )

            r2s = np.zeros(len(odmr_sample))
            for _idx, odmr in enumerate(odmr_sample.values()):
                r2s[_idx] = odmr.fit_model.rsquared
            min_r2 = np.min(r2s)

            if highest_min_r2 < min_r2:
                highest_min_r2 = min_r2
                optimal_params = (r2_thresh, thresh_frac, sigma_thresh_frac)

        return highest_min_r2, optimal_params

    @staticmethod
    def raster_odmr_fitting(
            odmr_measurements: dict[str, MeasurementDataclass],
            r2_thresh: float = 0.95,
            thresh_frac: float = 0.5,
            sigma_thresh_frac: float = 0.15,
            min_thresh: float = 0.01,
            extract_pixel_from_filename: bool = True,
            progress_bar: bool = True
    ) -> dict[str, MeasurementDataclass]:
        """
        Fit a list of ODMR data to single and double Lorentzian functions

        Args:
            odmr_measurements: List of ODMR data in MeasurementDataclasses
            r2_thresh: R^2 Threshold below which a double lorentzian is fitted instead of a single lorentzian
            thresh_frac:
            min_thresh:
            sigma_thresh_frac:
            extract_pixel_from_filename: Extract `(row, col)` (in this format) from filename
            progress_bar: Show progress bar
        Returns:
            List of ODMR data with fit, fit model and pixels in MeasurementDataclass
        """

        model1, base_params1 = rof.make_lorentzian_model()
        model2, base_params2 = rof.make_lorentziandouble_model()

        # Generate arguments for the parallel fitting
        args = []
        for odmr in tqdm(odmr_measurements.values(), disable=not progress_bar):
            x = odmr.data["Freq(MHz)"].to_numpy()
            y = odmr.data["Counts"].to_numpy()
            _, params1 = rof.estimate_lorentzian_dip(x, y, base_params1)
            _, params2 = rof.estimate_lorentziandouble_dip(x, y, base_params2, thresh_frac, min_thresh,
                                                           sigma_thresh_frac)
            args.append((x, y, model1, model2, params1, params2, r2_thresh))

        # Parallel fitting
        model_results = Parallel(n_jobs=cpu_count())(
            delayed(rof.lorentzian_fitting)(
                x, y, model1, model2, params1, params2, r2_thresh) for x, y, model1, model2, params1, params2, r2_thresh
            in
            tqdm(args, disable=not progress_bar)
        )

        x = list(odmr_measurements.values())[0].data["Freq(MHz)"].to_numpy()
        x_fit = np.linspace(start=x[0], stop=x[-1], num=int(len(x) * 2))

        for odmr, res in zip(odmr_measurements.values(), model_results):

            if len(res.params) == 6:
                # Fit to a single Lorentzian
                y_fit = model1.eval(x=x_fit, params=res.params)
            else:
                # Fit to a double Lorentzian
                y_fit = model2.eval(x=x_fit, params=res.params)

            # Plug results into the DataClass
            odmr.fit_model = res
            odmr.fit_data = pd.DataFrame(np.vstack((x_fit, y_fit)).T, columns=["x_fit", "y_fit"])

            if extract_pixel_from_filename:
                row, col = map(int, re.findall(r'(?<=\().*?(?=\))', odmr.filename)[0].split(","))
                odmr.xy_position = (row, col)

        return odmr_measurements

    @staticmethod
    def pixel_average_nan(orig_image: np.ndarray) -> np.ndarray:
        """ Average a nan pixel to its surrounding 8 points """
        image = orig_image.copy()
        for row, col in np.argwhere(np.isnan(image)):
            if row == 0:
                pixel_avg = np.nanmean(image[row + 1:row + 2, col - 1:col + 2])
            elif row == image.shape[0] - 1:
                pixel_avg = np.nanmean(image[row - 1:row, col - 1:col + 2])
            elif col == 0:
                pixel_avg = np.nanmean(image[row - 1:row + 2, col + 1:col + 2])
            elif col == image.shape[1] - 1:
                pixel_avg = np.nanmean(image[row - 1:row + 2, col - 1:col])
            else:
                pixel_avg = np.nanmean(image[row - 1:row + 2, col - 1:col + 2])

            image[row, col] = pixel_avg
        return image
