from __future__ import annotations

import datetime
import logging
import os
from typing import List, TYPE_CHECKING, Callable

from src.io_handler import IOHandler
from src.measurement_dataclass import RawTimetrace, PulsedMeasurement, PulsedMeasurementDataclass, \
    LaserPulses, MeasurementDataclass
from src.path_handler import PathHandler

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class DataLoaders(IOHandler):
    """
    Functions to use when importing/exporting data
    The first callable is the function for loading data
    The second callable is the function for loading params
    """
    default_loader: (Callable[[str], pd.DataFrame], Callable[[str], dict]) = (IOHandler.read_into_dataframe,
                                                                              IOHandler.read_qudi_parameters)
    confocal_loader: (Callable[[str, ...], np.ndarray], Callable[[str], dict]) = (IOHandler.read_into_ndarray,
                                                                                  IOHandler.read_qudi_parameters)
    trace_loader: (Callable[[str, ...], np.ndarray], Callable[[str], dict]) = (IOHandler.read_into_ndarray_transposed,
                                                                               IOHandler.read_qudi_parameters)
    nanonis_loader: (Callable[[str], pd.DataFrame], Callable[[str], dict]) = (IOHandler.read_nanonis_data,
                                                                              IOHandler.read_nanonis_parameters)
    figure_loader: Callable[[plt.Figure, str, ...], None] = IOHandler.savefig


class DataHandler(PathHandler, DataLoaders):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(__name__)

    def __load_pulsed_measurements_dataclass_list(self, pulsed_measurement_str: str) -> List[MeasurementDataclass]:
        measurement_filepaths, timestamps = [], []
        for filepath in self.get_measurement_filepaths(measurement="PulsedMeasurement", extension=".dat"):
            filename = os.path.basename(filepath)
            if pulsed_measurement_str in filename:
                timestamps.append(filename[:16])
                measurement_filepaths.append(filepath)

        pulsed_measurement_data: List[MeasurementDataclass] = []

        for timestamp in timestamps:
            pm, lp, rt = None, None, None
            for filepath in measurement_filepaths:
                filename = os.path.basename(filepath)
                if filename.startswith(timestamp):
                    if "laser_pulses" in filename:
                        lp = LaserPulses(filepath=filepath, loaders=DataLoaders.trace_loader)
                    elif "pulsed_measurement" in filename:
                        timestamp = datetime.datetime.strptime(filename[:16], "%Y%m%d-%H%M-%S"),
                        pm = PulsedMeasurement(filepath=filepath, loaders=DataLoaders.default_loader)
                    elif "raw_timetrace" in filename:
                        rt = RawTimetrace(filepath=filepath, loaders=DataLoaders.trace_loader)
                    if lp and pm and rt:
                        break
            pulsed_measurement_data.append(
                MeasurementDataclass(
                    timestamp=timestamp,
                    pulsed=PulsedMeasurementDataclass(
                        measurement=pm,
                        laser_pulses=lp,
                        timetrace=rt
                    )
                )
            )
        return pulsed_measurement_data

    def __load_standard_measurements_dataclass_list(self, standard_measurement_str: str) -> List[MeasurementDataclass]:
        standard_measurement_list: List[MeasurementDataclass] = []
        for filepath in self.get_measurement_filepaths(standard_measurement_str, extension=".dat"):
            try:
                timestamp = datetime.datetime.strptime(os.path.basename(filepath)[:16], "%Y%m%d-%H%M-%S")
            except ValueError as _:
                # Only raise warning once
                self.log.warning(f"Unable to extract timestamp from filepath {filepath}, using mtime instead")
                timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).astimezone()
            standard_measurement_list.append(
                MeasurementDataclass(
                    filepath=filepath,
                    timestamp=timestamp,
                    loaders=DataLoaders.default_loader
                )
            )
        return standard_measurement_list

    def load_measurements_into_dataclass_list(self, measurement_str: str) -> List[MeasurementDataclass]:
        if "PulsedMeasurement" in measurement_str:
            pulsed_measurement_str = measurement_str.split("/")[1]
            return self.__load_pulsed_measurements_dataclass_list(pulsed_measurement_str)
        else:
            return self.__load_standard_measurements_dataclass_list(measurement_str)

    @staticmethod
    def _extract_base_filepath(filepath: str) -> str:
        path_endswith = [
            "_pulsed_measurement",
            "_pulsed_measurement_fig",
            "_laser_pulses",
            "_raw_timetrace",
            "_fig"
        ]

        base_filepath = filepath
        for path in path_endswith:
            if filepath.endswith(path):
                base_filepath = filepath[:-len(path)]
                break
        return base_filepath

    @staticmethod
    def expand_timestamped_folderpath(filename: str):
        return os.path.join(filename[0:4], filename[4:6], filename[0:8])

    def load_measurement_list_into_dataclass_list(self, measurement_list: List[str]) -> List[MeasurementDataclass]:
        measurement_dataclass_list = []
        for measurement in measurement_list:
            if len(measurement.split("\\")) > 1:
                # Measurement performed from qudi-hira
                # Separate filename from measurement type
                measurement_type, filename = measurement.split("\\")

                # Recreate full filepath
                filepath = os.path.join(
                    self.data_folder_path,
                    self.expand_timestamped_folderpath(filename),
                    measurement_type,
                    filename
                )
                filepath = self._extract_base_filepath(filepath)
                timestamp = datetime.datetime.strptime(filename[:16], "%Y%m%d-%H%M-%S")

                if "PulsedMeasurement" in measurement:
                    measurement_dataclass_list.append(
                        MeasurementDataclass(
                            timestamp=timestamp,
                            filepath=filepath,
                            pulsed=PulsedMeasurementDataclass(
                                measurement=PulsedMeasurement(
                                    filepath=filepath + "_pulsed_measurement.dat",
                                    loaders=DataLoaders.default_loader
                                ),
                                laser_pulses=LaserPulses(
                                    filepath=filepath + "_laser_pulses.dat",
                                    loaders=DataLoaders.trace_loader
                                ),
                                timetrace=RawTimetrace(
                                    filepath=filepath + "_raw_timetrace.dat",
                                    loaders=DataLoaders.trace_loader
                                )
                            )
                        )
                    )
                else:
                    if "Confocal" in filepath:
                        loaders = DataLoaders.confocal_loader
                    else:
                        loaders = DataLoaders.default_loader
                    measurement_dataclass_list.append(
                        MeasurementDataclass(
                            filepath=filepath + ".dat",
                            timestamp=timestamp,
                            loaders=loaders
                        )
                    )
            else:
                # Generic measurement type
                if "frq-sweep" in measurement:
                    filepath = os.path.join(self.data_folder_path, measurement) + ".dat"
                    timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).astimezone()
                    measurement_dataclass_list.append(
                        MeasurementDataclass(
                            filepath=filepath,
                            timestamp=timestamp,
                            loaders=DataLoaders.nanonis_loader
                        )
                    )
        return measurement_dataclass_list

    def save_figures(self, fig: plt.Figure, filename: str, **kwargs):
        self.log.info(f"Saving '{filename}' to '{self.figure_folder_path}'")
        filepath = os.path.join(self.figure_folder_path, filename)
        DataLoaders.figure_loader(fig, filepath, **kwargs)
