from __future__ import annotations

import datetime
import logging
import os
from typing import List

import matplotlib.pyplot as plt

from src.measurement_dataclass import RawTimetrace, PulsedMeasurement, PulsedMeasurementDataclass, \
    LaserPulses, MeasurementDataclass
from src.path_handler import PathHandler

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class DataHandler(PathHandler):
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
                        lp = LaserPulses(filepath=filepath)
                    elif "pulsed_measurement" in filename:
                        timestamp = datetime.datetime.strptime(filename[:16], "%Y%m%d-%H%M-%S"),
                        pm = PulsedMeasurement(filepath=filepath)
                    elif "raw_timetrace" in filename:
                        rt = RawTimetrace(filepath=filepath)
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
            standard_measurement_list.append(MeasurementDataclass(filepath=filepath, timestamp=timestamp))
        return standard_measurement_list

    def load_measurements_into_dataclass_list(self, measurement_str: str) -> List[MeasurementDataclass]:
        if "PulsedMeasurement" in measurement_str:
            pulsed_measurement_str = measurement_str.split("/")[1]
            return self.__load_pulsed_measurements_dataclass_list(pulsed_measurement_str)
        else:
            return self.__load_standard_measurements_dataclass_list(measurement_str)

    def _extract_base_filepath(self, filepath: str) -> str:
        path_endswith = [
            "_pulsed_measurement",
            "_pulsed_measurement_fig",
            "_laser_pulses",
            "_raw_timetrace",
            "_fig"
        ]

        base_filepath = None
        for path in path_endswith:
            if filepath.endswith(path):
                base_filepath = filepath[:-len(path)]
                break
        if base_filepath is None:
            self.log.error(f"Unable to extract base filepath from {filepath}")
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
                                measurement=PulsedMeasurement(filepath=filepath + "_pulsed_measurement.dat"),
                                laser_pulses=LaserPulses(filepath=filepath + "_laser_pulses.dat"),
                                timetrace=RawTimetrace(filepath=filepath + "_raw_timetrace.dat")
                            )
                        )
                    )
                else:
                    measurement_dataclass_list.append(
                        MeasurementDataclass(
                            filepath=filepath + ".dat",
                            timestamp=timestamp
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
                            timestamp=timestamp
                        )
                    )
        return measurement_dataclass_list

    def save_figures(
            self,
            fig: plt.Figure,
            filename: str,
            overwrite: bool = True,
            only_jpg: bool = False,
            **kwargs
    ):
        """ Saves figures from matplotlib plot data. """

        if "." in filename:
            filename, _ = os.path.splitext(filename)

        self.log.info(f"Saving '{filename}' to '{self.figure_folder_path}'")

        if only_jpg:
            extensions = [".jpg"]
        else:
            extensions = [".jpg", ".pdf", ".svg", ".png"]

        for ext in extensions:
            figure_path = os.path.join(self.figure_folder_path, filename + ext)

            if not overwrite:
                if os.path.isfile(figure_path):
                    raise IOError(f"{figure_path} already exists")

            fig.savefig(figure_path, dpi=200, bbox_inches="tight", **kwargs)
