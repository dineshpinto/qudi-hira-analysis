from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import List, TYPE_CHECKING, Callable
from warnings import warn

from src.analysis_logic import AnalysisLogic
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
        self.analysis = AnalysisLogic()

        self.data_folder_path = self.__get_data_folder_path(data_folder, measurement_folder)
        if copy_measurement_folder_structure:
            self.figure_folder_path = self.__get_figure_folder_path(figure_folder, measurement_folder)
        else:
            self.figure_folder_path = figure_folder

        super().__init__(base_read_path=self.data_folder_path, base_write_path=self.figure_folder_path)

        self.timestamp_format_str = "%Y%m%d-%H%M-%S"

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
            if path.is_file() and measurement in str(path).lower() and exclude_str not in str(path):
                if extension:
                    if path.suffix == extension:
                        filepaths.append(path)
                else:
                    filepaths.append(path)
        return filepaths

    def __load_pulsed_measurements_dataclass_list(self, pulsed_measurement_str: str) -> dict[str: MeasurementDataclass]:
        filtered_filepaths = []
        timestamps = set()

        # Get set of unique timestamps containing pulsed_measurement_str
        for filepath in self.get_measurement_filepaths(measurement=pulsed_measurement_str):
            filename = filepath.name
            if pulsed_measurement_str in filename:
                timestamps.add(filename[:16])
                filtered_filepaths.append(filepath)

        pulsed_measurement_data: dict[str: MeasurementDataclass] = {}

        for ts in timestamps:
            pm, lp, rt = None, None, None

            for filepath in filtered_filepaths:
                filename = filepath.name
                if filename.startswith(ts):
                    if str(filename).endswith("laser_pulses.dat"):
                        lp = LaserPulses(filepath=filepath, loaders=self.trace_loader)
                    elif str(filename).endswith("pulsed_measurement.dat"):
                        pm = PulsedMeasurement(filepath=filepath, loaders=self.default_loader)
                    elif str(filename).endswith("raw_timetrace.dat"):
                        rt = RawTimetrace(filepath=filepath, loaders=self.trace_loader)

                if lp and pm and rt:
                    break

            pulsed_measurement_data[ts] = (
                MeasurementDataclass(
                    timestamp=datetime.datetime.strptime(ts, self.timestamp_format_str),
                    pulsed=PulsedMeasurementDataclass(
                        measurement=pm,
                        laser_pulses=lp,
                        timetrace=rt
                    )
                )
            )

        return pulsed_measurement_data

    def __load_standard_measurements_dataclass_list(self, standard_measurement_str: str) -> dict[
                                                                                            str: MeasurementDataclass]:
        standard_measurement_list: dict[str: MeasurementDataclass] = {}

        if standard_measurement_str.lower() == "confocal":
            loaders = self.confocal_loader
            exclude_str = "xy_data.dat"
        elif standard_measurement_str == "frq-sweep":
            loaders = self.nanonis_loader
            exclude_str: str = "image_1.dat"
        else:
            loaders = self.default_loader
            exclude_str: str = "image_1.dat"

        for filepath in self.get_measurement_filepaths(standard_measurement_str, extension=".dat",
                                                       exclude_str=exclude_str):
            ts = filepath.name[:16]
            standard_measurement_list[ts] = (
                MeasurementDataclass(
                    filepath=filepath,
                    timestamp=datetime.datetime.strptime(ts, "%Y%m%d-%H%M-%S"),
                    loaders=loaders
                )
            )
        return standard_measurement_list

    def load_measurements(self, measurement_str: str, pulsed: bool = False) -> dict[str: MeasurementDataclass]:
        measurement_str = measurement_str.lower()
        if pulsed:
            return self.__load_pulsed_measurements_dataclass_list(measurement_str)
        else:
            return self.__load_standard_measurements_dataclass_list(measurement_str)

    def load_measurements_into_dataclass_list(self, measurement_str: str) -> dict[str: MeasurementDataclass]:
        warn('This method is deprecated, "use load_measurements" instead', DeprecationWarning)
        if "PulsedMeasurement" in measurement_str:
            measurement_str = measurement_str.split("/")[1].lower()
            pulsed = True
        else:
            pulsed = False
        return self.load_measurements(measurement_str, pulsed=pulsed)
