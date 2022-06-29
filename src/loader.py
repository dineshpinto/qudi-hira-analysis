from __future__ import annotations

import os
from typing import List

from src.pathhandler import PathHandler
from src.measurement_dataclass import RawTimetrace, PulsedMeasurement, PulsedMeasurementData, \
    LaserPulses, MeasurementData


class DataLoader(PathHandler):
    def __init__(self, **kwargs):
        super(DataLoader, self).__init__(**kwargs)

    def __load_pulsed_measurement_dataclass_list(self, pulsed_measurement_str: str) -> List[MeasurementData]:
        measurement_filepaths, timestamps = [], []
        for filepath in self._get_measurement_filepaths(measurement="PulsedMeasurement", extension=".dat"):
            filename = os.path.basename(filepath)
            if pulsed_measurement_str in filename:
                timestamps.append(filename[:16])
                measurement_filepaths.append(filepath)

        pulsed_measurement_data: List[MeasurementData] = []

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
                MeasurementData(
                    filepath=pm.filepath,
                    pulsed_data=PulsedMeasurementData(
                        pulsed_measurement=pm,
                        laser_pulses=lp,
                        raw_timetrace=rt
                    )
                )
            )
        return pulsed_measurement_data

    def load_measurement_into_dataclass_list(self, measurement_str: str) -> List[MeasurementData]:
        if "PulsedMeasurement" in measurement_str:
            pulsed_measurement_str = measurement_str.split("/")[1]
            return self.__load_pulsed_measurement_dataclass_list(pulsed_measurement_str)
        else:
            measurement_list: List[MeasurementData] = []

            for filepath in self._get_measurement_filepaths(measurement_str, extension=".dat"):
                measurement_list.append(MeasurementData(filepath=filepath))

            return measurement_list
