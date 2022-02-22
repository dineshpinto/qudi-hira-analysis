import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Union, Tuple

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class PulsedMeasurement:
    filepath: str
    data: pd.DataFrame = field(repr=False)
    params: dict = field(repr=False)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    def get_data(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(self.data.columns) == 2:
            return (self.data["Controlled variable(s)"].to_numpy(),
                    self.data["Signal"].to_numpy())
        elif len(self.data.columns) == 3:
            return (self.data["Controlled variable(s)"].to_numpy(),
                    self.data["Signal"].to_numpy(),
                    self.data["Signal2"].to_numpy())
        else:
            raise ValueError(f"Invalid number of columns in DataFrame ({len(self.data.columns)})")


@dataclass
class LaserPulses:
    filepath: str
    data: np.ndarray = field(repr=False)
    params: dict = field(repr=False)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)


@dataclass
class RawTimetrace:
    filepath: str
    data: np.ndarray = field(repr=False)
    params: dict = field(repr=False)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)


@dataclass()
class PulsedData:
    timestamp: datetime.datetime
    pulsed_measurement: PulsedMeasurement
    laser_pulses: LaserPulses = field(default=None)
    raw_timetrace: RawTimetrace = field(default=None)

    def __post_init__(self):
        self.base_filename = self.pulsed_measurement.filename.replace("_pulsed_measurement.dat", "")

    def __repr__(self) -> str:
        return f"PulsedData(timestamp='{self.timestamp}', base_filename='{self.base_filename}')"

    def get_param_from_filename(self, unit: str = "dBm") -> float:
        return float(re.findall("([0-9]+)" + f"{unit}", self.pulsed_measurement.filename)[0])

    def show_image(self) -> Image:
        return Image.open(self.pulsed_measurement.filepath.replace(".dat", "_fig.png"))
