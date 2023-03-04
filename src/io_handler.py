import ast
import datetime
import inspect
import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _check_extension(filepath: Path, extension: str) -> Path:
    """
    Check if filepath extension matches expected extension.
    - If no extension found, add extension
    - If wrong extension found, raise IOError
    """
    if filepath.suffix == extension:
        return filepath
    elif filepath.suffix == "":
        return filepath.with_suffix(extension)
    else:
        raise IOError(f"Invalid extension '{filepath.suffix}' in '{filepath}', extension should be '{extension}'")


class IOHandler:
    """ Handle all read and write operations. """

    def __init__(self, base_read_path: Path = None, base_write_path: Path = None):
        self.base_read_path = base_read_path
        self.base_write_path = base_write_path

    def read_qudi_parameters(self, filepath: Path) -> dict:
        """ Extract parameters from a qudi dat file. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
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

    def read_into_dataframe(self, filepath: Path) -> pd.DataFrame:
        """ Read a qudi data file into a pd DataFrame for analysis. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        filepath = _check_extension(filepath, ".dat")

        with open(filepath) as handle:
            # Generate column names for DataFrame by parsing the file
            *_comments, names = itertools.takewhile(lambda line: line.startswith('#'), handle)
            names = names[1:].strip().split("\t")
        return pd.read_csv(filepath, names=names, comment="#", sep="\t")

    def read_csv(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """ Read a csv file into a pd DataFrame. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath

        return pd.read_csv(filepath, **kwargs)

    def read_confocal_into_dataframe(self, filepath: Path) -> pd.DataFrame:
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        filepath = _check_extension(filepath, ".dat")

        confocal_params = self.read_qudi_parameters(filepath)
        data = self.read_into_ndarray(filepath, delimiter="\t")

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

    def read_into_ndarray(self, filepath: Path, **kwargs) -> np.ndarray:
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        return np.genfromtxt(filepath, **kwargs)

    def read_into_ndarray_transposed(self, filepath: Path, **kwargs) -> np.ndarray:
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        return np.genfromtxt(filepath, **kwargs).T

    def read_pys(self, filepath: Path) -> np.ndarray:
        """ Loads raw pys data files. Wraps around numpy.load. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        filepath = _check_extension(filepath, ".pys")
        return np.load(filepath, encoding="bytes", allow_pickle=True)

    def read_pkl(self, filepath: Path) -> dict:
        """ Loads processed pickle files for plotting/further analysis. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        filepath = _check_extension(filepath, ".pys")

        with open(filepath, 'rb') as f:
            file = pickle.load(f)
        return file

    def read_nanonis_data(self, filepath: Path) -> pd.DataFrame:
        """ Extract data from a Nanonis dat file. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
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

    def read_nanonis_parameters(self, filepath: Path) -> dict:
        """ Extract parameters from a Nanonis dat file. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
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

    def read_pfeiffer_data(self, filepath: Path) -> pd.DataFrame:
        """ Read data stored by Pfeiffer vacuum monitoring software. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
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

    def read_lakeshore_data(self, filepath: Path) -> pd.DataFrame:
        """ Read data stored by Lakeshore temperature monitor software. """
        if self.base_read_path:
            filepath = self.base_read_path / filepath
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
        if self.base_read_path:
            filepath = self.base_read_path / filepath
        filepath = _check_extension(filepath, ".txt")

        df = pd.read_csv(filepath, sep="\t", skiprows=14, names=["wavelength", "intensity"])
        return df

    @staticmethod
    def __get_forward_backward_counts(count_rates, num_pixels):
        split_array = np.split(count_rates, 2 * num_pixels)

        # Extract forward scan array as every second element
        forward_counts = np.stack(split_array[::2])
        # Extract backward scan array as every shifted second element
        # Flip scan so that backward and forward scans represent the same data
        backward_counts = np.flip(np.stack(split_array[1::2]), axis=1)
        return forward_counts, backward_counts

    def read_pixelscanner_data(self, filepath: Path) -> (np.ndarray, np.ndarray):
        df = self.read_into_dataframe(filepath)

        num_pixels = int(np.sqrt(len(df) // 2))
        if num_pixels ** 2 != len(df) // 2:
            raise ValueError("Number of pixels does not match data length.")

        try:
            forward, backward = self.__get_forward_backward_counts(df["count_rates"], num_pixels)
        except KeyError:
            # Support old data format
            forward = df["forward (cps)"].to_numpy().reshape(num_pixels, num_pixels)
            backward = df["backward (cps)"].to_numpy().reshape(num_pixels, num_pixels)
        return forward, backward

    def save_pkl(self, obj: object, filepath: Path):
        """ Saves processed pickle files for plotting/further analysis. """
        if self.base_write_path:
            filepath = self.base_write_path / filepath
        filepath.parent.mkdir(exist_ok=True)
        filepath = _check_extension(filepath, ".pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    def save_pys(self, dictionary: dict, filepath: Path):
        """ Saves processed pickle files for plotting/further analysis. """
        if self.base_write_path:
            filepath = self.base_write_path / filepath
        filepath.parent.mkdir(exist_ok=True)
        filepath = _check_extension(filepath, ".pys")

        with open(filepath, 'wb') as f:
            pickle.dump(dictionary, f, 1)

    def save_df(self, df: pd.DataFrame, filepath: Path):
        """ Save Dataframe as csv. """
        if self.base_write_path:
            filepath = self.base_write_path / filepath
        filepath.parent.mkdir(exist_ok=True)
        filepath = _check_extension(filepath, ".pys")

        df.to_csv(filepath, sep='\t', encoding='utf-8')

    def save_figures(self, fig: plt.Figure, filepath: Path, **kwargs):
        """
        Saves figures from matplotlib plot data. By default, saves as jpg, png, pdf and svg.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filepath : pathlib.Path
            Path to save figure to. If called with DataHandler, only the filename is required.

        Keyword Arguments
        -----------------
        only_jpg : bool
            If True, only save as jpg. Default is False.
        only_pdf : bool
            If True, only save as pdf. Default is False.
        **kwargs
            Keyword arguments passed to fig.savefig().
        """
        if self.base_write_path:
            filepath = self.base_write_path / filepath
        filepath.parent.mkdir(exist_ok=True)

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
            fig.savefig(filepath.with_suffix(ext), dpi=200, **kwargs)
