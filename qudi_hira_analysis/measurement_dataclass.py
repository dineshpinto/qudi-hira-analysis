from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    import datetime
    from pathlib import Path

    import lmfit
    import numpy as np

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s',
                    level=logging.INFO)


@dataclass
class PulsedMeasurement:
    """ Dataclass for storing pulsed data and metadata """
    filepath: Path
    loaders: (Callable, Callable) = field(default=None)
    __data: pd.DataFrame = field(default=None)
    __params: dict = field(default=None)

    @property
    def data(self) -> pd.DataFrame:
        """ Read measurement data from file into pandas DataFrame """
        if self.__data is None:
            self.__data = self.loaders[0](self.filepath)
        return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            self.__params = self.loaders[1](self.filepath)
        return self.__params


@dataclass
class LaserPulses:
    """ Dataclass for storing laser pulses data and metadata """
    filepath: Path
    loaders: (Callable, Callable) = field(default=None)
    __data: np.ndarray = field(default=None)
    __params: dict = field(default=None)

    @property
    def data(self) -> np.ndarray:
        """ Read measurement data from file into pandas DataFrame """
        if self.__data is None:
            self.__data = self.loaders[0](self.filepath)
        return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            self.__params = self.loaders[1](self.filepath)
        return self.__params


@dataclass
class RawTimetrace:
    """ Dataclass for storing raw timetrace data and metadata """
    filepath: Path
    loaders: (Callable, Callable) = field(default=None)
    __data: np.ndarray = field(default=None)
    __params: dict = field(default=None)

    @property
    def data(self) -> np.ndarray:
        """ Read measurement data from file into pandas DataFrame """
        if self.__data is None:
            self.__data = self.loaders[0](self.filepath)
        return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            self.__params = self.loaders[1](self.filepath)
        return self.__params


@dataclass
class PulsedMeasurementDataclass:
    """ Dataclass for storing pulsed measurement data and metadata """
    measurement: PulsedMeasurement
    laser_pulses: LaserPulses = field(default=None)
    timetrace: RawTimetrace = field(default=None)

    def __post_init__(self):
        self.base_filename = self.measurement.filepath.name.replace(
            "_pulsed_measurement.dat", "")

    def show_image(self) -> Image:
        """ Use PIL to open the measurement image saved on the disk """
        return Image.open(
            str(self.measurement.filepath).replace(".dat","_fig.png"))


@dataclass
class MeasurementDataclass:
    """ Dataclass for storing measurement data and metadata """
    timestamp: datetime.datetime
    filepath: Path = field(default=None)
    _loaders: (Callable, Callable) = field(default=None)
    pulsed: PulsedMeasurementDataclass = field(default=None)
    __data: np.ndarray | pd.DataFrame = field(default=None)
    __params: dict = field(default=None)
    _fit_data: pd.DataFrame = field(default=None)
    _fit_model: lmfit.Model = field(default=None)
    _xy_position: tuple[int, int] = field(default=None)

    def __post_init__(self):
        self.log = logging.getLogger(__name__)

        if self.pulsed:
            self.filename = self.pulsed.base_filename
        else:
            self.filename = self.filepath.name

    def __repr__(self) -> str:
        return (f"MeasurementDataclass(timestamp='{self.timestamp}', "
                f"filename='{self.filename}')")

    @property
    def data(self) -> np.ndarray | pd.DataFrame:
        """ Read measurement data from file. """
        if self.pulsed:
            return self.pulsed.measurement.data
        else:
            if self.__data is None:
                self.__data = self._loaders[0](self.filepath)
            return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file. """
        if self.pulsed:
            return self.pulsed.measurement.params
        else:
            if self.__params is None:
                self.__params = self._loaders[1](self.filepath)
            return self.__params

    @property
    def fit_data(self) -> pd.DataFrame:
        """ Fit data to a model """
        return self._fit_data

    @fit_data.setter
    def fit_data(self, fit_data: pd.DataFrame):
        """ Fit data to a model """
        self._fit_data = fit_data

    @property
    def fit_model(self) -> lmfit.Model:
        """ lmfit data model """
        return self._fit_model

    @fit_model.setter
    def fit_model(self, fit_model: lmfit.Model):
        self._fit_model = fit_model

    @property
    def xy_position(self) -> tuple[int, int]:
        """ (row, col) position of measurement in image """
        return self._xy_position

    @xy_position.setter
    def xy_position(self, xy_position: tuple[int, int]):
        self._xy_position = xy_position

    def get_param_from_filename(self, unit: str) -> float | None:
        """
        Extract param from filename with format <param><unit>, where param
        is a float or integer and unit is a string. The param can be negative
        with keyword 'minus' or a decimal point with keyword 'point'.

        Args:
            unit: unit of param to extract, e.g. dBm, mbar, V etc.

        Returns:
            Extracted param from filename

        Examples:
            filename = "rabi_12dBm"
            >>> get_param_from_filename(unit='dBm')
            12.0

            filename = "pixelscan_minus100nm"
            >>> get_param_from_filename(unit='dBm')
            -100.0

            filename = "rabi_2e-6mbar"
            >>> get_param_from_filename(unit='mbar')
            2e-6

            filename = "rabi_2point3uW"
            >>> get_param_from_filename(unit='uW')
            2.5
        """
        filename = self.filename
        filename = filename.replace("point", ".")
        filename = filename.replace("minus", "-")
        params = re.search(rf"(-?\d+\.?\d*)(?={unit})", filename)

        if params:
            # Handle exponents in filename
            if filename[params.start() - 1] == "e":
                try:
                    params = re.search(
                        rf"(-?_\d)[^a]+?(?={unit})", filename
                    ).group(0)[1:]
                    return float(params)
                except AttributeError as exc:
                    raise Exception(
                        f"Parameter with unit '{unit}' not found in "
                        f"filename '{filename}'"
                    ) from exc
            else:
                return float(params.group(0))
        else:
            self.log.warning(f"Unable to extract parameter from filename: {filename}")
            return None

    def set_datetime_index(self) -> pd.DataFrame:
        if 'Start counting time' not in self.__params:
            raise ValueError("'Start counting time' not in params")
        if not isinstance(self.__data, pd.DataFrame):
            raise TypeError("data is not of type pd.DataFrame")
        if "Time (s)" not in self.__data.columns:
            raise IndexError("Unable to find column 'Time (s)' in DataFrame")

        self.__data['Time (s)'] += self.__params['Start counting time'].timestamp()
        self.__data["Time"] = pd.to_datetime(self.__data['Time (s)'], unit='s',
                                             utc=True)
        self.__data.set_index(self.__data["Time"], inplace=True)
        self.__data.tz_convert('Europe/Berlin')
        self.__data.drop(["Time", "Time (s)"], inplace=True, axis=1)
        return self.__data
