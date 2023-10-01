from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from qudi_hira_analysis.analysis_logic import AnalysisLogic
from qudi_hira_analysis.io_handler import IOHandler
from qudi_hira_analysis.measurement_dataclass import (
    LaserPulses,
    MeasurementDataclass,
    PulsedMeasurement,
    PulsedMeasurementDataclass,
    RawTimetrace,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import pySPM

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s',
                    level=logging.INFO)


class DataLoader(IOHandler):
    """
    Interface to map measurement data loading methods in IOHandler to the automated data
    methods in DataHandler. Also provides a direct passthrough of the IOHandler methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create callables used in measurement dataclasses
        self.default_qudi_loader: (
            Callable[[Path], pd.DataFrame], Callable[[Path], dict]) = (
            self.read_into_dataframe,
            self.read_qudi_parameters
        )
        self.confocal_qudi_loader: (
            Callable[[Path], np.ndarray], Callable[[Path], dict]) = (
            self.read_confocal_into_dataframe,
            self.read_qudi_parameters
        )
        self.pixelscanner_qudi_loader: (
            Callable[[Path], (pySPM.SPM_image, pySPM.SPM_image)],
            Callable[[Path], dict]) = (
            self.read_pixelscanner_data,
            self.read_qudi_parameters
        )
        self.trace_qudi_loader: (
            Callable[[Path], np.ndarray], Callable[[Path], dict]) = (
            self.read_into_ndarray_transposed,
            self.read_qudi_parameters
        )
        self.nanonis_loader: (
            Callable[[Path], pd.DataFrame], Callable[[Path], dict]) = (
            self.read_nanonis_data,
            self.read_nanonis_parameters
        )
        self.nanonis_spm_loader: (Callable[[Path], pySPM.SXM], None) = (
            self.read_nanonis_spm_data,
            None
        )
        self.bruker_spm_loader: (Callable[[Path], pySPM.Bruker], None) = (
            self.read_bruker_spm_data,
            None
        )
        self.temperature_loader: (Callable[[Path], pd.DataFrame], None) = (
            self.read_lakeshore_data,
            None
        )
        self.pys_loader: (Callable[[Path], dict], None) = (
            self.read_pys,
            None
        )
        self.pressure_loader: (Callable[[Path], pd.DataFrame], None) = (
            self.read_pfeiffer_data,
            None
        )


class DataHandler(DataLoader, AnalysisLogic):
    """
    Handles automated data searching and extraction into dataclasses.

    Args:
        data_folder: Path to the data folder.
        figure_folder: Path to the figure folder.
        measurement_folder: Path to the measurement folder.

    Examples:
        Create an instance of the DataHandler class:

        >>> dh = DataHandler(
        >>>     data_folder=Path('C:\\'', 'Data'),
        >>>     figure_folder=Path('C:\\'', 'QudiHiraAnalysis'),
        >>>     measurement_folder=Path('20230101_Bakeout'),
        >>> )
    """

    def __init__(
            self,
            data_folder: Path,
            figure_folder: Path,
            measurement_folder: Path = Path(),
            copy_measurement_folder_structure: bool = True,
    ):
        self.log = logging.getLogger(__name__)

        self.data_folder_path = self.__get_data_folder_path(data_folder,
                                                            measurement_folder)

        if copy_measurement_folder_structure:
            self.figure_folder_path = self.__get_figure_folder_path(figure_folder,
                                                                    measurement_folder)
        else:
            self.figure_folder_path = self.__get_figure_folder_path(figure_folder,
                                                                    Path())

        super().__init__(base_read_path=self.data_folder_path,
                         base_write_path=self.figure_folder_path)

        self.timestamp_format_str = "%Y%m%d-%H%M-%S"

    def __get_data_folder_path(self, data_folder: Path, folder_name: Path) -> Path:
        """ Check if folder exists, if not, create and return absolute folder paths. """
        path = data_folder / folder_name

        if not path.exists():
            raise OSError("Data folder path does not exist.")

        self.log.info(f"Data folder path is {path}")
        return path

    def __get_figure_folder_path(self, figure_folder: Path, folder_name: Path) -> Path:
        """ Check if folder exists, if not, create and return absolute folder paths. """
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

    def __print_or_return_tree(self, folder: Path, print_tree: bool) -> str | None:
        """ Print or return a tree of the data and figure folders. """
        if print_tree:
            for line in self.__tree(folder):
                print(line)
        else:
            tree = ""
            for line in self.__tree(folder):
                tree += line + "\n"
            return tree

    def data_folder_tree(self, print_tree: bool = True) -> str | None:
        """
        Print or return a string tree of the data folder.

        Args:
            print_tree: Print the tree or return it as a string (default: True).

        Returns:
            str: The tree as a string if `print_tree is False.
        """
        return self.__print_or_return_tree(self.data_folder_path, print_tree=print_tree)

    def figure_folder_tree(self, print_tree: bool = True) -> str | None:
        """
        Print or return a string tree of the figure folder.

        Args:
            print_tree: Print the tree or return it as a string (default: True).

        Returns:
            str: The tree as a string if `print_tree is False.
        """
        return self.__print_or_return_tree(self.figure_folder_path,
                                           print_tree=print_tree)

    def _get_measurement_filepaths(
            self,
            measurement: str,
            extension: str,
            exclude_str: str | None = None
    ) -> list[Path]:
        """
        List all measurement files for a single measurement type, regardless of date
        within a similar set (i.e. top level folder).
        """
        filepaths: list[Path] = []

        for path in self.data_folder_path.rglob("*"):
            if (
                    path.is_file() and
                    measurement.lower() in str(path).lower() and
                    path.suffix == extension and
                    (exclude_str is None or exclude_str not in str(path))
            ):
                filepaths.append(path)
        return filepaths

    def __load_qudi_pulsed_measurement_into_dataclass(
            self,
            measurement_str: str,
            extension: str
    ) -> dict[str: MeasurementDataclass]:
        """ Load all qudi pulsed measurements into a dictionary of dataclasses. """
        filtered_filepaths = []
        timestamps = set()

        # Get set of unique timestamps containing pulsed_measurement_str
        for filepath in self._get_measurement_filepaths(measurement=measurement_str,
                                                        extension=extension,
                                                        exclude_str="image_1.dat"):
            timestamps.add(filepath.name[:16])
            filtered_filepaths.append(filepath)

        pulsed_measurement_data: dict[str: MeasurementDataclass] = {}

        for idx, ts in enumerate(timestamps):
            pm, lp, rt = None, None, None

            for filepath in filtered_filepaths:
                filename = filepath.name
                if filename.startswith(ts):
                    if str(filename).endswith("laser_pulses.dat"):
                        lp = LaserPulses(filepath=filepath,
                                         loaders=self.trace_qudi_loader)
                    elif str(filename).endswith("pulsed_measurement.dat"):
                        pm = PulsedMeasurement(filepath=filepath,
                                               loaders=self.default_qudi_loader)
                    elif str(filename).endswith("raw_timetrace.dat"):
                        rt = RawTimetrace(filepath=filepath,
                                          loaders=self.trace_qudi_loader)

                if lp and pm and rt:
                    break

            if not (lp and pm and rt):
                raise OSError(
                    f"'{filtered_filepaths[idx]}' is a invalid pulsed measurement.")

            pulsed_measurement_data[ts] = (
                MeasurementDataclass(
                    timestamp=datetime.datetime.strptime(
                        ts,
                        self.timestamp_format_str
                    ),
                    pulsed=PulsedMeasurementDataclass(
                        measurement=pm,
                        laser_pulses=lp,
                        timetrace=rt
                    )
                )
            )
        return pulsed_measurement_data

    def __load_qudi_measurements_into_dataclass(
            self,
            measurement_str: str,
            extension: str
    ) -> dict[str: MeasurementDataclass]:
        """ Load all qudi measurements into a dictionary of dataclasses. """
        if measurement_str.lower() == "confocal":
            loaders = self.confocal_qudi_loader
            exclude_str = "xy_data.dat"
        elif measurement_str.lower() == "pixelscanner":
            loaders = self.pixelscanner_qudi_loader
            exclude_str = None
        else:
            loaders = self.default_qudi_loader
            exclude_str = None

        measurement_data: dict[str: MeasurementDataclass] = {}

        for filepath in self._get_measurement_filepaths(measurement_str, extension,
                                                        exclude_str):
            ts = filepath.name[:16]
            measurement_data[ts] = (
                MeasurementDataclass(
                    filepath=filepath,
                    timestamp=datetime.datetime.strptime(
                        ts,
                        self.timestamp_format_str
                    ),
                    _loaders=loaders
                )
            )
        return measurement_data

    def __load_standard_measurements_into_dataclass(
            self,
            measurement_str: str,
            extension: str
    ) -> dict[str: MeasurementDataclass]:
        """ Load all standard measurements into a dictionary of dataclasses. """
        measurement_list: dict[str: MeasurementDataclass] = {}

        # Try and infer measurement type
        if measurement_str.lower() == "temperature-monitoring":
            loaders = self.temperature_loader
            extension = ".xls"
            exclude_str = None
        elif measurement_str.lower() == "pressure-monitoring":
            loaders = self.pressure_loader
            extension = ".txt"
            exclude_str = None
        elif measurement_str == "frq-sweep":
            loaders = self.nanonis_loader
            exclude_str = None
        elif extension == ".sxm":
            loaders = self.nanonis_spm_loader
            exclude_str = None
        elif extension == ".pys":
            loaders = self.pys_loader
            exclude_str = None
        elif extension == ".001":
            loaders = self.bruker_spm_loader
            exclude_str = None
        else:
            loaders = self.default_qudi_loader
            exclude_str = None

        for filepath in self._get_measurement_filepaths(measurement_str, extension,
                                                        exclude_str):
            timestamp = datetime.datetime.fromtimestamp(filepath.stat().st_mtime)
            self.log.warning(
                "Extracting timestamp from file modified time, may not be accurate.")
            ts = datetime.datetime.strftime(timestamp, self.timestamp_format_str)
            measurement_list[ts] = (
                MeasurementDataclass(
                    filepath=filepath,
                    timestamp=timestamp,
                    _loaders=loaders
                )
            )
        return measurement_list

    def load_measurements(
            self,
            measurement_str: str,
            qudi: bool = True,
            pulsed: bool = False,
            extension: str = ".dat"
    ) -> dict[str: MeasurementDataclass]:
        """
        Lazy-load all measurements of a given type into a dictionary of dataclasses.

        Args:
            measurement_str: The name of the measurement type to load e.g. t1, t2,
                             confocal etc.
                             Recursively searches through the path defined by
                             data_folder and measurement_folder.
                             Case-insensitive (i.e. "odmr" == "ODMR" == "Odmr").
            qudi: Whether the measurement is a qudi measurement (default: True).
            pulsed: Whether the measurement is a pulsed measurement (default: False).
            extension: The file extension of the measurement files (default: .dat).

        Returns:
            dict: A dictionary where keys are the measurement timestamps and values are
                  dataclasses containing the measurement data.

        Examples:
            `dh` is an instance of the `DataHandler` class.

            Load all ODMR measurements:

            >>> dh.load_measurements(measurement_str="ODMR", pulsed=True)

            Load all confocal data:

            >>> dh.load_measurements(measurement_str="Confocal")

            Load all temperature monitoring data:

            >>> dh.load_measurements(measurement_str="Temperature")

            Load all pressure monitoring data:

            >>> dh.load_measurements(measurement_str="Pressure")
        """

        measurement_str = measurement_str.lower()
        if qudi:
            if pulsed:
                return self.__load_qudi_pulsed_measurement_into_dataclass(
                    measurement_str, extension=".dat"
                )
            else:
                return self.__load_qudi_measurements_into_dataclass(
                    measurement_str, extension=".dat"
                )
        else:
            return self.__load_standard_measurements_into_dataclass(
                measurement_str, extension=extension
            )
