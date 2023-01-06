import ast
import datetime
import inspect
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _check_extension(filepath: str, extension: str) -> str:
    """
    Check if filepath extension matches expected extension.
    - If no extension found, add extension
    - If wrong extension found, raise IOError
    """
    filename = os.path.basename(filepath)

    if filename.endswith(extension):
        return filepath
    elif "." in filename:
        _, ext = filename.split(".")
        raise IOError(f"Invalid extension '{ext}' in '{filename}', extension should be '{extension}'")
    else:
        filepath += extension
    return filepath


def _channel_to_gauge_names(channel_names: list) -> list:
    """ Replace the channel names with gauge locations. """
    gauges = {"CH 1": "Main", "CH 2": "Prep", "CH 3": "Backing"}
    try:
        return [gauges[ch] for ch in channel_names]
    except KeyError:
        return channel_names


class IOHandler:
    """ Handle all read and write operations. """

    def __init__(self, base_read_path: str = "", base_write_path: str = ""):
        self.base_read_path = base_read_path
        self.base_write_path = base_write_path

    def read_qudi_parameters(self, filepath: str) -> dict:
        """ Extract parameters from a qudi dat file. """
        filepath = os.path.join(self.base_read_path, filepath)
        filepath = _check_extension(filepath, ".dat")

        params = {}
        with open(filepath) as file:
            for line in file:
                if line == '#=====\n':
                    break
                else:
                    # noinspection PyBroadException
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

    def read_into_dataframe(self, filepath: str) -> pd.DataFrame:
        """ Read a qudi data file into a pd DataFrame for analysis. """
        filepath = os.path.join(self.base_read_path, filepath)

        with open(filepath) as handle:
            # Generate column names for DataFrame by parsing the file
            *_comments, names = itertools.takewhile(lambda line: line.startswith('#'), handle)
            names = names[1:].strip().split("\t")
        return pd.read_csv(filepath, names=names, comment="#", sep="\t")

    def read_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """ Read a csv file into a pd DataFrame. """
        filepath = os.path.join(self.base_read_path, filepath)
        return pd.read_csv(filepath, **kwargs)

    def read_confocal_into_dataframe(self, filepath) -> pd.DataFrame:
        filepath = os.path.join(self.base_read_path, filepath)

        confocal_params = IOHandler.read_qudi_parameters(filepath)
        data = IOHandler.read_into_ndarray(filepath)

        # Use the confocal parameters to generate the index and columns for the DataFrame
        index = np.linspace(
            confocal_params['X image min (m)'],
            confocal_params['X image max (m)'],
            data.shape[0]
        )
        columns = np.linspace(
            confocal_params['Y image min'],
            confocal_params['Y image max'],
            data.shape[1]
        )
        df = pd.DataFrame(data, index=index, columns=columns)
        # Sort the index to get origin (0, 0) in the lower left corner of the DataFrame
        df.sort_index(axis=0, ascending=False, inplace=True)
        return df

    def read_into_ndarray(self, filepath: str, **kwargs) -> np.ndarray:
        filepath = os.path.join(self.base_read_path, filepath)
        return np.genfromtxt(filepath, **kwargs)

    def read_into_ndarray_transposed(self, filepath: str, **kwargs) -> np.ndarray:
        filepath = os.path.join(self.base_read_path, filepath)
        return np.genfromtxt(filepath, **kwargs).T

    def read_pys(self, filepath: str) -> np.ndarray:
        """ Loads raw pys data files. Wraps around numpy.load. """
        filepath = os.path.join(self.base_read_path, filepath)

        filepath = _check_extension(filepath, ".pys")
        return np.load(filepath, encoding="bytes", allow_pickle=True)

    def save_pys(self, dictionary: dict, filepath: str):
        """ Saves processed pickle files for plotting/further analysis. """
        filepath = os.path.join(self.base_write_path, filepath)

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = _check_extension(filepath, ".pys")

        with open(filepath, 'wb') as f:
            pickle.dump(dictionary, f, 1)

    def save_df(self, df: pd.DataFrame, filepath: str):
        """ Save Dataframe as csv. """
        filepath = os.path.join(self.base_write_path, filepath)

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = _check_extension(filepath, ".csv")

        df.to_csv(filepath, sep='\t', encoding='utf-8')

    def read_pkl(self, filepath: str) -> dict:
        """ Loads processed pickle files for plotting/further analysis. """
        filepath = os.path.join(self.base_read_path, filepath)
        filepath = _check_extension(filepath, ".pkl")

        with open(filepath, 'rb') as f:
            file = pickle.load(f)
        return file

    def save_pkl(self, obj: object, filepath: str):
        """ Saves processed pickle files for plotting/further analysis. """
        filepath = os.path.join(self.base_write_path, filepath)

        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = _check_extension(filepath, ".pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    def read_nanonis_data(self, filepath: str) -> pd.DataFrame:
        """ Extract data from a Nanonis dat file. """
        filepath = os.path.join(self.base_read_path, filepath)

        filepath = _check_extension(filepath, ".dat")

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

    def read_nanonis_parameters(self, filepath: str) -> dict:
        """ Extract parameters from a Nanonis dat file. """
        filepath = os.path.join(self.base_read_path, filepath)

        filepath = _check_extension(filepath, ".dat")

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
                        # Convert strings to number where possible
                        value = float(value)
                    except ValueError:
                        pass
                    if "Oscillation Control>" in label:
                        label = label.replace("Oscillation Control>", "")
                    parameters[label] = value
        return parameters

    def read_pfeiffer_data(self, filepath: str) -> pd.DataFrame:
        """ Read data stored by Pfeiffer vacuum monitoring software. """
        filepath = os.path.join(self.base_read_path, filepath)

        filepath = _check_extension(filepath, ".txt")

        # Extract rows including the header
        df = pd.read_csv(filepath, sep="\t", skiprows=[0, 2, 3, 4])
        # Combine data and time columns together
        df["Date"] = df["Date"] + " " + df["Time"]
        df = df.drop("Time", axis=1)

        # Infer datetime format and convert to datetime objects
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

        # Set datetime as index
        df = df.set_index("Date", drop=True)

        return df

    def read_lakeshore_data(self, filepath: str) -> pd.DataFrame:
        """ Read data stored by Lakeshore temperature monitor software. """
        filepath = os.path.join(self.base_read_path, filepath)

        filepath = _check_extension(filepath, ".xls")

        # Extract only the origin timestamp
        origin = pd.read_excel(filepath, skiprows=1, nrows=1, usecols=[1], header=None)[1][0]
        # Remove any tzinfo to prevent future exceptions in pandas
        origin = origin.replace("CET", "")
        # Parse datetime object from timestamp
        origin = pd.to_datetime(origin)

        # Create DataFrame and drop empty cols
        df = pd.read_excel(filepath, skiprows=3)
        df = df.dropna(axis=1, how="all")
        # Add datetimes to DataFrame
        df["Datetime"] = pd.to_datetime(df["Time"], unit="ms", origin=origin)
        df = df.drop("Time", axis=1)
        # Set datetime as index
        df = df.set_index("Datetime", drop=True)
        return df

    def read_oceanoptics_data(self, filepath: str) -> pd.DataFrame:
        """ Read spectrometer data from OceanOptics spectrometer. """
        filepath = os.path.join(self.base_read_path, filepath)

        filepath = _check_extension(filepath, ".txt")

        df = pd.read_csv(filepath, sep="\t", skiprows=14, names=["wavelength", "intensity"])
        return df

    def save_figures(self, fig: plt.Figure, filepath: str, **kwargs):
        """ Saves figures from matplotlib plot data. """
        filepath = os.path.join(self.base_write_path, filepath)

        extensions = None
        if "only_jpg" in kwargs:
            if kwargs.get("only_jpg"):
                extensions = [".jpg"]
            kwargs.pop("only_jpg", None)
        elif "only_pdf" in kwargs:
            if kwargs.get("only_pdf"):
                extensions = [".pdf"]
            kwargs.pop("only_pdf", None)
        else:
            extensions = [".jpg", ".pdf", ".svg", ".png"]

        for ext in extensions:
            figure_path = filepath + ext
            fig.savefig(figure_path, dpi=200, **kwargs)
