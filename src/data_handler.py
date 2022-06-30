from __future__ import annotations

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
        super(DataHandler, self).__init__(**kwargs)
        self.log = logging.getLogger(__name__)

    def __load_pulsed_measurements_dataclass_list(self, pulsed_measurement_str: str) -> List[MeasurementDataclass]:
        measurement_filepaths, timestamps = [], []
        for filepath in self._get_measurement_filepaths(measurement="PulsedMeasurement", extension=".dat"):
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
                        pm = PulsedMeasurement(filepath=filepath)
                    elif "raw_timetrace" in filename:
                        rt = RawTimetrace(filepath=filepath)
                    if lp and pm and rt:
                        break
            pulsed_measurement_data.append(
                MeasurementDataclass(
                    filepath=pm.filepath,
                    pulsed_data=PulsedMeasurementDataclass(
                        pulsed_measurement=pm,
                        laser_pulses=lp,
                        raw_timetrace=rt
                    )
                )
            )
        return pulsed_measurement_data

    def load_measurements_into_dataclass_list(self, measurement_str: str) -> List[MeasurementDataclass]:
        if "PulsedMeasurement" in measurement_str:
            pulsed_measurement_str = measurement_str.split("/")[1]
            return self.__load_pulsed_measurements_dataclass_list(pulsed_measurement_str)
        else:
            measurement_list: List[MeasurementDataclass] = []

            for filepath in self._get_measurement_filepaths(measurement_str, extension=".dat"):
                measurement_list.append(MeasurementDataclass(filepath=filepath))

            return measurement_list

    def save_figures(self, fig: plt.Figure, filename: str, overwrite: bool = True, only_jpg: bool = False):
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

            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
