from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple

import pandas as pd

from src.analysis_logic import AnalysisLogic
from src.io_handler import IOHandler

if TYPE_CHECKING:
    import datetime
    import numpy as np
    from PIL import Image
    from lmfit.model import ModelResult


@dataclass
class PulsedMeasurement(IOHandler):
    filepath: str
    __data: pd.DataFrame = field(default=None)
    __params: dict = field(default=None)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    @property
    def data(self) -> pd.DataFrame:
        """ Read measurement data from file into pandas DataFrame """
        if self.__data is None:
            self.__data = self.read_into_dataframe(self.filepath)
        return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            self.__params = self.read_qudi_parameters(self.filepath)
        return self.__params


@dataclass
class LaserPulses(IOHandler):
    filepath: str
    __data: np.ndarray = field(default=None)
    __params: dict = field(default=None)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    @property
    def data(self) -> np.ndarray:
        """ Read measurement data from file into pandas DataFrame """
        if self.__data is None:
            self.__data = self.read_into_ndarray(self.filepath).T
        return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            self.__params = self.read_qudi_parameters(self.filepath)
        return self.__params


@dataclass
class RawTimetrace(IOHandler):
    filepath: str
    __data: np.ndarray = field(default=None)
    __params: dict = field(default=None)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    @property
    def data(self) -> np.ndarray:
        """ Read measurement data from file into pandas DataFrame """
        if self.__data.size is None:
            self.__data = self.read_into_ndarray(self.filepath).T
        return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            self.__params = self.read_qudi_parameters(self.filepath)
        return self.__params


@dataclass()
class PulsedMeasurementDataclass:
    measurement: PulsedMeasurement
    laser_pulses: LaserPulses = field(default=None)
    timetrace: RawTimetrace = field(default=None)

    def __post_init__(self):
        self.base_filename = self.measurement.filename.replace("_pulsed_measurement.dat", "")

    def show_image(self) -> Image:
        """ Use PIL to open the measurement image saved on the disk """
        return Image.open(self.measurement.filepath.replace(".dat", "_fig.png"))


@dataclass
class MeasurementDataclass(IOHandler):
    timestamp: datetime.datetime
    filepath: str = field(default=None)
    pulsed: PulsedMeasurementDataclass = field(default=None)
    __data: np.ndarray | pd.DataFrame = field(default=None)
    __params: dict = field(default=None)
    fit_result: ModelResult = field(default=None)

    def __post_init__(self):
        super().__init__()
        self.analysis = AnalysisLogic()
        self.filename = os.path.basename(self.filepath)

    def __repr__(self) -> str:
        return f"Measurement(timestamp='{self.timestamp}', filename='{self.filename}')"

    @property
    def data(self) -> np.ndarray | pd.DataFrame:
        """ Read measurement data from file into suitable data structure """
        if self.pulsed:
            return self.pulsed.measurement.data
        else:
            if self.__data is None:
                # Add custom measurement loading logic here
                if "Confocal" in self.filepath:
                    self.__data = self.read_into_ndarray(self.filepath, dtype=int, delimiter='\t')
                elif "frq-sweep" in self.filepath:
                    self.__data = self._extract_data_from_nanonis_dat(self.filepath)
                else:
                    self.__data = self.read_into_dataframe(self.filepath)
            return self.__data

    @property
    def params(self) -> dict:
        """ Read measurement params from file into dict """
        if self.__params is None:
            # Add custom parameter loading logic here
            if "frq-sweep" in self.filepath:
                self.__params = self._extract_parameters_from_nanonis_dat(self.filepath)
            else:
                self.__params = self.read_qudi_parameters(self.filepath)
        return self.__params

    def get_param_from_filename(self, unit: str = "dBm") -> float:
        """ Extract param from filename with format <param><unit>, example 12dBm -> 12 """
        params = re.findall("(-?\d+\.?\d*)" + f"{unit}", self.filename)
        if len(params) == 0:
            raise ValueError(f"Parameter with unit '{unit}' not found in filename '{self.filename}'")
        else:
            return float(params[0])

    def set_datetime_index(self) -> pd.DataFrame:
        if 'Start counting time' not in self.__params:
            raise ValueError("'Start counting time' not in params")
        if not isinstance(self.__data, pd.DataFrame):
            raise TypeError("data is not of type pd.DataFrame")
        if "Time (s)" not in self.__data.columns:
            raise IndexError("Unable to fine 'Time (s)' in DataFrame")

        self.__data['Time (s)'] += self.__params['Start counting time'].timestamp()
        self.__data["Time"] = pd.to_datetime(self.__data['Time (s)'], unit='s', utc=True)
        self.__data.set_index(self.__data["Time"], inplace=True)
        self.__data.tz_convert('Europe/Berlin')
        self.__data.drop(["Time", "Time (s)"], inplace=True, axis=1)
        return self.__data

    def fit(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """ Wrapper around AnalysisLogic.perform_fit() to populate fit_result attribute. """
        fit_x, fit_y, self.fit_result = self.analysis.perform_fit(**kwargs)
        return fit_x, fit_y
