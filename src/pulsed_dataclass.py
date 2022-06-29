"""
The following code is part of qudiamond-analysis under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2022 Dinesh Pinto. See the LICENSE file at the
top-level directory of this distribution and at
<https://github.com/dineshpinto/qudiamond-analysis/>
"""

import datetime
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image

if TYPE_CHECKING:
    from src.io import IO


@dataclass()
class PulsedMeasurement:
    io: IO
    filepath: str
    data: pd.DataFrame = field(default=pd.DataFrame)

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    def get_data(self) -> pd.DataFrame:
        """ Read measurement data from file into pandas DataFrame """
        if self.data.empty:
            self.data = self.io.read_into_df(self.filepath)
        return self.data

    def get_params(self) -> dict:
        """ Read measurement params from file into dict """
        return self.io.read_qudi_parameters(self.filepath)


@dataclass()
class LaserPulses:
    io: IO
    filepath: str
    data: np.ndarray = field(default=np.array([]))

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    def get_data(self) -> np.ndarray:
        """ Read measurement data from file into pandas DataFrame """
        if self.data.size == 0:
            self.data = np.genfromtxt(self.filepath).T
        return self.data

    def get_params(self) -> dict:
        """ Read measurement params from file into dict """
        return self.io.read_qudi_parameters(self.filepath)


@dataclass()
class RawTimetrace:
    io: IO
    filepath: str
    data: np.ndarray = field(default=np.array([]))

    def __post_init__(self):
        self.filename = os.path.basename(self.filepath)

    def get_data(self) -> np.ndarray:
        """ Read measurement data from file into pandas DataFrame """
        if self.data.size == 0:
            self.data = np.genfromtxt(self.filepath).T
        return self.data

    def get_params(self) -> dict:
        """ Read measurement params from file into dict """
        return self.io.read_qudi_parameters(self.filepath)


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
        """ Extract param from filename with format <param><unit>, example 12dBm -> 12 """
        return float(re.findall("(-?\d+\.?\d*)" + f"{unit}", self.pulsed_measurement.filename)[0])

    def show_image(self) -> Image:
        """ Use PIL to open the measurement image saved on the disk """
        return Image.open(self.pulsed_measurement.filepath.replace(".dat", "_fig.png"))
