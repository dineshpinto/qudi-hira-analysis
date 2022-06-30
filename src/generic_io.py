import ast
import datetime
import inspect
import itertools
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from dateutil import parser


class GenericIO:
    """ Read and write files (eg. pys, df, pkl) """

    @staticmethod
    def read_qudi_parameters(filepath: str) -> dict:
        """ Extract parameters from a qudi dat file. """
        if not filepath.endswith(".dat"):
            filepath += ".dat"

        params = {}
        with open(filepath) as file:
            for line in file:
                if line == '#=====\n':
                    break
                else:
                    try:
                        # Remove # from beginning of lines
                        line = line[1:]
                        if line.count(":") == 1:
                            # Add params to dictionary
                            label, value = line.split(":")
                            if value != "\n":
                                params[label] = ast.literal_eval(inspect.cleandoc(value))
                        elif line.count(":") == 3:
                            # Handle files with timestamps in them
                            label = line.split(":")[0]
                            timestamp_str = "".join(line.split(":")[1:]).strip()
                            datetime_str = datetime.datetime.strptime(timestamp_str, "%d.%m.%Y %Hh%Mmin%Ss").replace(
                                tzinfo=datetime.timezone.utc)
                            params[label] = datetime_str
                    except Exception as _:
                        pass
        return params

    @staticmethod
    def read_into_dataframe(filepath: str) -> pd.DataFrame:
        """ Read a qudi data file into a pd DataFrame for analysis. """
        with open(filepath) as handle:
            # Generate column names for DataFrame by parsing the file
            *_comments, names = itertools.takewhile(lambda line: line.startswith('#'), handle)
            names = names[1:].strip().split("\t")
        return pd.read_csv(filepath, names=names, comment="#", sep="\t")

    @staticmethod
    def read_into_ndarray(filepath: str, **kwargs) -> np.ndarray:
        return np.genfromtxt(filepath, **kwargs)

    @staticmethod
    def load_pys(filepath: str) -> np.ndarray:
        """ Loads raw pys data files. Wraps around numpy.load. """
        if not filepath.endswith(".pys"):
            filepath += ".pys"
        return np.load(filepath, encoding="bytes", allow_pickle=True)

    @staticmethod
    def save_pys(dictionary: dict, filepath: str):
        """ Saves processed pickle files for plotting/further analysis. """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if not filepath.endswith(".pys"):
            filepath += ".pys"

        with open(filepath, 'wb') as f:
            pickle.dump(dictionary, f, 1)

    @staticmethod
    def save_df(df: pd.DataFrame, filepath: str):
        """ Save Dataframe as csv. """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if not filepath.endswith(".csv"):
            filepath += ".csv"

        df.to_csv(filepath, sep='\t', encoding='utf-8')

    @staticmethod
    def load_pkl(filepath: str) -> dict:
        """ Loads processed pickle files for plotting/further analysis. """
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"

        with open(filepath, 'rb') as f:
            file = pickle.load(f)
        return file

    @staticmethod
    def save_pkl(obj: object, filepath: str):
        """ Saves processed pickle files for plotting/further analysis. """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if not filepath.endswith(".pkl"):
            filepath += ".pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def _extract_data_from_nanonis_dat(filepath: str) -> pd.DataFrame:
        """ Extract data from a Nanonis dat file. """
        if not filepath.endswith(".dat"):
            filepath += ".dat"

        skip_rows = 0
        with open(filepath) as dat_file:
            for num, line in enumerate(dat_file, 1):
                if "[DATA]" in line:
                    # Find number of rows to skip when extracting data
                    skip_rows = num
                    break
                if "#=====" in line:
                    skip_rows = num
                    break

        df = pd.read_table(filepath, sep="\t", skiprows=skip_rows)
        return df

    @staticmethod
    def _extract_parameters_from_nanonis_dat(filepath: str) -> dict:
        """ Extract parameters from a Nanonis dat file. """
        if not filepath.endswith(".dat"):
            filepath += ".dat"

        parameters = {}
        with open(filepath) as dat_file:
            for line in dat_file:
                if line == "\n":
                    # Break when reaching empty line
                    break
                elif "User" in line or line.split("\t")[0] == "":
                    # Cleanup excess parameters and skip empty lines
                    pass
                else:
                    label, value, _ = line.split("\t")
                    try:
                        # Convert strings to numbers where possible
                        value = float(value)
                    except ValueError:
                        pass
                    if "Oscillation Control>" in label:
                        label = label.replace("Oscillation Control>", "")
                    parameters[label] = value
        return parameters

    def read_nanonis_data(self, filepath: str) -> Tuple[dict, pd.DataFrame]:
        """ Convenience function to extract both parameters and data from a DAT file. """
        parameters = self._extract_parameters_from_nanonis_dat(filepath)
        data = self._extract_data_from_nanonis_dat(filepath)
        return parameters, data

    @staticmethod
    def _channel_to_gauge_names(channel_names: list) -> list:
        """ Replace the channel names with gauge locations. """
        gauges = {"CH 1": "Main", "CH 2": "Prep", "CH 3": "Backing"}
        return [gauges.get(ch, ch) for ch in channel_names]

    @staticmethod
    def _convert_tpg_to_mpl_time(df: pd.DataFrame) -> np.ndarray:
        """
        Read DataFrame extracted using read_tpg_data and add in matplotlib
        datetimes using "Date" and "Time" cols.
        """
        datetimes = df["Date"] + " " + df["Time"]
        # Convert raw dates and times to datetime Series, then to an matplotlib Series
        dt_series_datetime = [datetime.datetime.strptime(str(dt), '%d-%b-%y %H:%M:%S.%f') for dt in datetimes]
        # noinspection PyUnresolvedReferences
        dt_series_mpl = matplotlib.dates.date2num(dt_series_datetime)
        return dt_series_mpl

    def read_pfeiffer_data(self, filepath: str) -> pd.DataFrame:
        """ Read data stored by Pfeiffer vacuum monitoring software. """
        if not filepath.endswith(".txt"):
            filepath += ".txt"

        # Extract only the header to check which gauges are connected
        file_header = pd.read_csv(filepath, sep="\t", skiprows=1, nrows=1)
        header = self._channel_to_gauge_names(file_header)

        # Create DataFrame with new header
        df = pd.read_csv(filepath, sep="\t", skiprows=5, names=header)

        # Save matplotlib datetimes for plotting
        df["MPL_datetimes"] = self._convert_tpg_to_mpl_time(df)
        return df

    @staticmethod
    def read_lakeshore_data(filepath: str) -> pd.DataFrame:
        """ Read data stored by Lakeshore temperature monitor software. """
        if not filepath.endswith(".xls"):
            filepath += ".xls"

        # Extract only the timestamp
        timestamp = pd.read_excel(filepath, skiprows=1, nrows=1, usecols=[1], header=None)[1][0]
        # Parse datetime object from timestamp
        timestamp_dt = parser.parse(timestamp, tzinfos={"CET": 0 * 3600})

        # Create DataFrame
        df = pd.read_excel(filepath, skiprows=3)

        # Add matplotlib datetimes to DataFrame
        time_array = []
        for milliseconds in df["Time"]:
            time_array.append(timestamp_dt + datetime.timedelta(milliseconds=milliseconds))
        # noinspection PyUnresolvedReferences
        df["MPL_datetimes"] = matplotlib.dates.date2num(time_array)
        return df

    @staticmethod
    def read_oceanoptics_data(filepath: str) -> pd.DataFrame:
        """ Read spectrometer data from OceanOptics spectrometer. """
        if not filepath.endswith(".txt"):
            filepath += ".txt"

        df = pd.read_csv(filepath, sep="\t", skiprows=14, names=["wavelength", "intensity"])
        return df
