from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import List, TYPE_CHECKING, Callable

from src.io_handler import IOHandler
from src.measurement_dataclass import RawTimetrace, PulsedMeasurement, PulsedMeasurementDataclass, \
    LaserPulses, MeasurementDataclass

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class DataHandler(IOHandler):
    def __init__(
            self,
            data_folder: Path,
            figure_folder: Path,
            measurement_folder: Path,
            copy_measurement_folder_structure: bool = True
    ):
        """
        This class is specific for qudi-hira measurements. To support adding new measurement types to the dataclass
        the callable needs to be added into `DataLoader` and corresponding functions need to be added into
        `DataHandler`.
        :param data_folder: Path to the base data folder
        :param figure_folder: Path where figures should be saved
        :param measurement_folder: Path to the specific measurement folder inside data folder
        """
        self.log = logging.getLogger(__name__)
        self.data_folder_path = self.__get_data_folder_path(data_folder, measurement_folder)
        if copy_measurement_folder_structure:
            self.figure_folder_path = self.__get_figure_folder_path(figure_folder, measurement_folder)
        else:
            self.figure_folder_path = figure_folder

        super().__init__(base_read_path=self.data_folder_path, base_write_path=self.figure_folder_path)

        # Create callables used in measurement dataclasses
        self.default_loader: (Callable[[Path], pd.DataFrame], Callable[[Path], dict]) = (
            self.read_into_dataframe,
            self.read_qudi_parameters
        )
        self.confocal_loader: (Callable[[Path, ...], np.ndarray], Callable[[Path], dict]) = (
            self.read_confocal_into_dataframe,
            self.read_qudi_parameters
        )
        self.trace_loader: (Callable[[Path, ...], np.ndarray], Callable[[Path], dict]) = (
            self.read_into_ndarray_transposed,
            self.read_qudi_parameters
        )
        self.nanonis_loader: (Callable[[Path], pd.DataFrame], Callable[[Path], dict]) = (
            self.read_nanonis_data,
            self.read_nanonis_parameters
        )
        self.figure_loader: Callable[[plt.Figure, Path, ...], None] = self.save_figures

    def __get_data_folder_path(self, data_folder: Path, folder_name: Path) -> Path:
        """ Check if folder exists and return absolute folder paths. """
        path = data_folder / folder_name

        if not path.exists():
            raise IOError("Data folder path does not exist.")

        self.log.info(f"Data folder path is {path}")
        return path

    def __get_figure_folder_path(self, figure_folder: Path, folder_name: Path) -> Path:
        """ Check if folder exists, if not, create it and return absolute folder paths. """
        path = figure_folder / folder_name

        if not path.exists():
            path.mkdir()
            self.log.info(f"Creating new output folder path {path}")
        else:
            self.log.info(f"Figure folder path is {path}")
        return path

    def __tree(self, dir_path: Path, prefix: str = ''):
        """
        A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        # prefix components:
        space = '    '
        branch = '│   '
        # pointers:
        tee = '├── '
        last = '└── '

        contents = list(dir_path.iterdir())
        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            yield prefix + pointer + path.name
            if path.is_dir():  # extend the prefix and recurse:
                extension = branch if pointer == tee else space
                # i.e. space because last, └── , above so no more |
                yield from self.__tree(path, prefix=prefix + extension)

    def data_folder_tree(self):
        """ Print a tree of the data folder. """
        for line in self.__tree(self.data_folder_path):
            print(line)

    def figure_folder_tree(self):
        """ Print a tree of the figure folder. """
        for line in self.__tree(self.figure_folder_path):
            print(line)

    def get_measurement_filepaths(
            self,
            measurement: str,
            extension: str = ".dat",
            exclude_str: str = "image_1.dat"
    ) -> list:
        """
        List all measurement files for a single measurement type, regardless of date
        within a similar set (i.e. top level folder).
        """
        filepaths: List[Path] = []

        for path in self.data_folder_path.rglob("*"):
            if path.is_file() and measurement in str(path) and exclude_str not in str(path):
                if extension:
                    if path.suffix == extension:
                        filepaths.append(path)
                else:
                    filepaths.append(path)
        return filepaths

    def __load_pulsed_measurements_dataclass_list(self, pulsed_measurement_str: str) -> List[MeasurementDataclass]:
        measurement_filepaths, timestamps = [], []
        for filepath in self.get_measurement_filepaths(measurement="PulsedMeasurement", extension=".dat"):
            filename = filepath.name
            if pulsed_measurement_str in filename:
                timestamps.append(filename[:16])
                measurement_filepaths.append(filepath)

        pulsed_measurement_data: List[MeasurementDataclass] = []

        for timestamp in timestamps:
            pm, lp, rt = None, None, None
            for filepath in measurement_filepaths:
                filename = filepath.name
                if filename.startswith(timestamp):
                    if "laser_pulses" in filename:
                        lp = LaserPulses(filepath=filepath, loaders=self.trace_loader)
                    elif "pulsed_measurement" in filename:
                        timestamp = datetime.datetime.strptime(filename[:16], "%Y%m%d-%H%M-%S"),
                        pm = PulsedMeasurement(filepath=filepath, loaders=self.default_loader)
                    elif "raw_timetrace" in filename:
                        rt = RawTimetrace(filepath=filepath, loaders=self.trace_loader)
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

            if "Confocal" in filepath:
                loaders = self.confocal_loader
            elif "frq-sweep" in filepath:
                loaders = self.nanonis_loader
            else:
                loaders = self.default_loader

            standard_measurement_list.append(
                MeasurementDataclass(
                    filepath=filepath,
                    timestamp=timestamp,
                    loaders=loaders
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
                                    loaders=self.default_loader
                                ),
                                laser_pulses=LaserPulses(
                                    filepath=filepath + "_laser_pulses.dat",
                                    loaders=self.trace_loader
                                ),
                                timetrace=RawTimetrace(
                                    filepath=filepath + "_raw_timetrace.dat",
                                    loaders=self.trace_loader
                                )
                            )
                        )
                    )
                else:
                    if "Confocal" in filepath:
                        loaders = self.confocal_loader
                    else:
                        loaders = self.default_loader
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
                            loaders=self.nanonis_loader
                        )
                    )
        return measurement_dataclass_list
