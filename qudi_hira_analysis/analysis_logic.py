from __future__ import annotations

import logging
import random
import re
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

import qudi_hira_analysis._raster_odmr_fitting as rof
from qudi_hira_analysis._qudi_fit_logic import FitLogic

if TYPE_CHECKING:
    from lmfit import Model, Parameter, Parameters
    from lmfit.model import ModelResult

    from .measurement_dataclass import MeasurementDataclass

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s',
                    level=logging.INFO)


class FitMethodsAndEstimators:
    """
        Class for storing fit methods and estimators.
        Fit methods are stored as tuples of (method, estimator)
        where method is the name of the fit method and estimator is the name of the
        estimator.


        The fit functions available are:

        | Dimension | Fit                           |
        |-----------|-------------------------------|
        | 1D        | decayexponential              |
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
        | 2D        | twoDgaussian                  |
    """
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
    twoDgaussian: tuple = ("twoDgaussian", "generic")  # noqa: N815


class AnalysisLogic(FitLogic):
    """ Class for performing analysis on measurement data """
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
            parameters: list[Parameter] | None = None,
            dims: str = "1d") -> tuple[np.ndarray, np.ndarray, ModelResult]:
        fit = {
            dims: {'default': {'fit_function': fit_function, 'estimator': estimator}}}
        user_fit = self.validate_load_fits(fit)

        if parameters:
            user_fit[dims]["default"]["parameters"].add_many(*parameters)

        use_settings = {}
        for key in user_fit[dims]["default"]["parameters"]:
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
            parameters: list[Parameter] | None = None
    ) -> tuple[np.ndarray, np.ndarray, ModelResult]:
        """
        Args:
            x: x data, can be string, numpy array or pandas Series
            y: y data, can be string, numpy array or pandas Series
            fit_function: fit function to use
            data: pandas DataFrame containing x and y data, if None x and y must be
            numpy arrays or pandas Series
            parameters: list of parameters to use in fit (optional)

        Returns:
            Fit x data, fit y data and lmfit ModelResult
        """
        if "twoD" in fit_function[0]:
            dims: str = "2d"
        else:
            dims: str = "1d"

        if data is None:
            if isinstance(x, (pd.Series, pd.Index)):
                x: np.ndarray = x.to_numpy()
            if isinstance(y, pd.Series):
                y: np.ndarray = y.to_numpy()
        elif isinstance(data, pd.DataFrame):
            x: np.ndarray = data[x].to_numpy()
            y: np.ndarray = data[y].to_numpy()
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

    def get_all_fits(self) -> tuple[list, list]:
        """Get all available fits

        Returns:
            Tuple with list of 1d and 2d fits
        """
        one_d_fits: list = list(self.fit_list['1d'].keys())
        two_d_fits: list = list(self.fit_list['2d'].keys())
        self.log.info(f"1d fits: {one_d_fits}\n2d fits: {two_d_fits}")
        return one_d_fits, two_d_fits

    @staticmethod
    def analyze_mean(
            laser_data: np.ndarray,
            signal_start: float = 100e-9,
            signal_end: float = 300e-9,
            bin_width: float = 1e-9
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the mean of the signal window.

        Args:
            laser_data: 2D array of laser data
            signal_start: start of the signal window in seconds
            signal_end: end of the signal window in seconds
            bin_width: width of a bin in seconds

        Returns:
            Mean of the signal window and measurement error
        """
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
    def analyze_mean_reference(
            laser_data: np.ndarray,
            signal_start: float = 100e-9,
            signal_end: float = 300e-9,
            norm_start: float = 1000e-9,
            norm_end: float = 2000e-9,
            bin_width: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        """
        Subtracts the mean of the signal window from the mean of the reference window.

        Args:
            laser_data: 2D array of laser data
            signal_start: start of the signal window in seconds
            signal_end: end of the signal window in seconds
            norm_start: start of the reference window in seconds
            norm_end: end of the reference window in seconds
            bin_width: width of a bin in seconds

        Returns:
            Referenced mean of the signal window and measurement error
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
            counts = laser_arr[norm_start_bin:norm_end_bin]
            reference_sum = np.sum(counts)
            reference_mean = (reference_sum / len(counts)) if len(counts) != 0 else 0.0

            # calculate the sum and mean of the data in the signal window
            counts = laser_arr[signal_start_bin:signal_end_bin]
            signal_sum = np.sum(counts)
            signal_mean = (signal_sum / len(counts)) if len(counts) != 0 else 0.0

            signal_data[ii] = signal_mean - reference_mean

            # calculate with respect to gaussian error 'evolution'
            error_data[ii] = signal_data[ii] * np.sqrt(
                1 / abs(signal_sum) + 1 / abs(reference_sum))

        return signal_data, error_data

    @staticmethod
    def analyze_mean_norm(
            laser_data: np.ndarray,
            signal_start: float = 100e-9,
            signal_end: float = 300e-9,
            norm_start: float = 1000e-9,
            norm_end=2000e-9,
            bin_width: float = 1e-9
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Divides the mean of the signal window from the mean of the reference window.

        Args:
            laser_data: 2D array of laser data
            signal_start: start of the signal window in seconds
            signal_end: end of the signal window in seconds
            norm_start: start of the reference window in seconds
            norm_end: end of the reference window in seconds
            bin_width: width of a bin in seconds

        Returns:
            Normalized mean of the signal window and measurement error
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
            counts = laser_arr[norm_start_bin:norm_end_bin]
            reference_sum = np.sum(counts)
            reference_mean = (reference_sum / len(counts)) if len(counts) != 0 else 0.0

            # calculate the sum and mean of the data in the signal window
            counts = laser_arr[signal_start_bin:signal_end_bin]
            signal_sum = np.sum(counts)
            signal_mean = (signal_sum / len(counts)) if len(counts) != 0 else 0.0

            # Calculate normalized signal while avoiding division by zero
            if reference_mean > 0 and signal_mean >= 0:
                signal_data[ii] = signal_mean / reference_mean
            else:
                signal_data[ii] = 0.0

            # Calculate measurement error while avoiding division by zero
            if reference_sum > 0 and signal_sum > 0:
                # calculate with respect to gaussian error 'evolution'
                error_data[ii] = signal_data[ii] * np.sqrt(
                    1 / signal_sum + 1 / reference_sum)
            else:
                error_data[ii] = 0.0

        return signal_data, error_data

    def optimize_raster_odmr_params(
            self,
            measurements: dict[str, MeasurementDataclass],
            num_samples: int = 10,
            num_params: int = 3,
    ) -> tuple[float, tuple[float, float, float]]:
        """
        This method optimizes the hyperparameters of the ODMR analysis.
        It does so by randomly sampling a subset of the measurements and
        then optimizing the hyperparameters for them.

        Args:
            measurements: A dictionary of measurements to optimize the hyperparameters.
            num_params: The number of parameters to optimize.
            num_samples: The number of measurements to sample.

        Returns:
            The highest minimum R2 value and the optimized hyperparameters.
        """
        r2_threshs: np.ndarray = np.around(
            np.linspace(start=0.9, stop=0.99, num=num_params),
            decimals=2
        )
        thresh_fracs: np.ndarray = np.around(
            np.linspace(start=0.5, stop=0.9, num=num_params),
            decimals=1
        )
        sigma_thresh_fracs: np.ndarray = np.around(
            np.linspace(start=0.1, stop=0.2, num=num_params),
            decimals=1
        )

        odmr_sample: dict = {}
        for k, v in random.sample(sorted(measurements.items()), k=num_samples):
            odmr_sample[k] = v

        highest_min_r2: float = 0
        optimal_params: tuple[float, float, float] = (0, 0, 0)

        for r2_thresh, thresh_frac, sigma_thresh_frac in product(
                r2_threshs, thresh_fracs, sigma_thresh_fracs):
            odmr_sample = self.fit_raster_odmr(
                odmr_sample,
                r2_thresh=r2_thresh,
                thresh_frac=thresh_frac,
                sigma_thresh_frac=sigma_thresh_frac,
                min_thresh=0.01,
                progress_bar=False
            )

            r2s: np.ndarray = np.zeros(len(odmr_sample))
            for idx, odmr in enumerate(odmr_sample.values()):
                r2s[idx] = odmr.fit_model.rsquared
            min_r2: float = np.min(r2s)

            if highest_min_r2 < min_r2:
                highest_min_r2 = min_r2
                optimal_params = (r2_thresh, thresh_frac, sigma_thresh_frac)

        return highest_min_r2, optimal_params

    @staticmethod
    def _lorentzian_fitting(
            x: np.ndarray,
            y: np.ndarray,
            model1: Model,
            model2: Model,
            params1: Parameters,
            params2: Parameters,
            r2_thresh: float
    ) -> ModelResult:
        """ Make Lorentzian fitting for single and double Lorentzian model """
        res1 = rof.make_lorentzian_fit(x, y, model1, params1)
        if res1.rsquared < r2_thresh:
            return rof.make_lorentziandouble_fit(x, y, model2, params2)
        return res1

    def fit_raster_odmr(
            self,
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
            odmr_measurements: Dict of ODMR data in MeasurementDataclasses
            r2_thresh: R^2 Threshold below which a double lorentzian is fitted instead
                of a single lorentzian
            thresh_frac: Threshold fraction for the peak finding
            min_thresh: Minimum threshold for the peak finding
            sigma_thresh_frac: Change in threshold fraction for the peak finding
            extract_pixel_from_filename: Extract `(row, col)` (in this format) from
                filename
            progress_bar: Show progress bar

        Returns:
            Dict of ODMR MeasurementDataclass with fit, fit model and pixels attributes
            set
        """

        model1, base_params1 = rof.make_lorentzian_model()
        model2, base_params2 = rof.make_lorentziandouble_model()

        # Generate arguments for the parallel fitting
        args = []
        for odmr in tqdm(odmr_measurements.values(), disable=not progress_bar):
            x = odmr.data["Freq(MHz)"].to_numpy()
            y = odmr.data["Counts"].to_numpy()
            _, params1 = rof.estimate_lorentzian_dip(x, y, base_params1)
            _, params2 = rof.estimate_lorentziandouble_dip(
                x, y, base_params2, thresh_frac, min_thresh, sigma_thresh_frac
            )
            args.append((x, y, model1, model2, params1, params2, r2_thresh))

        # Parallel fitting
        model_results = Parallel(n_jobs=cpu_count())(
            delayed(self._lorentzian_fitting)(
                x, y, model1, model2, params1, params2, r2_thresh) for
            x, y, model1, model2, params1, params2, r2_thresh
            in
            tqdm(args, disable=not progress_bar)
        )

        x = next(iter(odmr_measurements.values())).data["Freq(MHz)"].to_numpy()
        x_fit = np.linspace(start=x[0], stop=x[-1], num=int(len(x) * 2))

        for odmr, res in zip(odmr_measurements.values(), model_results):

            if len(res.params) == 6:
                # Evaluate a single Lorentzian
                y_fit = model1.eval(x=x_fit, params=res.params)
            else:
                # Evaluate a double Lorentzian
                y_fit = model2.eval(x=x_fit, params=res.params)

            # Plug results into the DataClass
            odmr.fit_model = res
            odmr.fit_data = pd.DataFrame(np.vstack((x_fit, y_fit)).T,
                                         columns=["x_fit", "y_fit"])

            if extract_pixel_from_filename:
                # Extract the pixel with regex from the filename
                row, col = map(
                    int,
                    re.findall(r'(?<=\().*?(?=\))', odmr.filename)[0].split(",")
                )
                odmr.xy_position = (row, col)

        return odmr_measurements

    @staticmethod
    def average_raster_odmr_pixels(orig_image: np.ndarray) -> np.ndarray:
        """ Average a NaN pixel to its surrounding pixels.

        Args:
            orig_image: Image with NaN pixels

        Returns:
            Image with NaN pixels replaced by the average of its surrounding pixels
        """
        image: np.ndarray = orig_image.copy()
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
