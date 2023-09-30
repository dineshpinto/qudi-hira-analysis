import ast
import contextlib
import datetime
import inspect
import itertools
import pickle
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pySPM


class IOHandler:
    """ Handle all read and write operations. """

    def __init__(
            self,
            base_read_path: Optional[Path] = None,
            base_write_path: Optional[Path] = None
    ):
        super().__init__()
        self.base_read_path = base_read_path
        self.base_write_path = base_write_path

    @staticmethod
    def _add_base_read_path(func: Callable) -> Callable:
        """
        Decorator to add the `base_read_path` to the filepath if it is not None

        Args:
            func: Function to be decorated

        Returns:
            Decorated function
        """

        @wraps(func)
        def wrapper(self, filepath: Path, **kwargs):
            if self.base_read_path:
                filepath = self.base_read_path / filepath
            return func(self, filepath, **kwargs)

        return wrapper

    @staticmethod
    def _add_base_write_path(func: Callable) -> Callable:
        """
        Decorator to add the `base_write_path` to the filepath if it is not None

        Args:
            func: Function to be decorated

        Returns:
            Decorated function
        """

        @wraps(func)
        def wrapper(self, filepath: Path, **kwargs):
            if self.base_write_path:
                filepath = self.base_write_path / filepath
            filepath.parent.mkdir(exist_ok=True)
            return func(self, filepath, **kwargs)

        return wrapper

    @staticmethod
    def _check_extension(ext: str) -> Callable:
        """
        Decorator to check the extension of the filepath is correct

        Args:
            ext: Extension to check for

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self, filepath: Path, **kwargs) -> Callable:
                if filepath.suffix == ext:
                    return func(self, filepath, **kwargs)
                elif filepath.suffix == "":
                    return func(self, filepath.with_suffix(ext), **kwargs)
                else:
                    raise OSError(
                        f"Invalid extension '{filepath.suffix}' in '{filepath}', "
                        f"extension should be '{ext}'")

            return wrapper

        return decorator

    @_add_base_read_path
    @_check_extension(".dat")
    def read_qudi_parameters(self, filepath: Path) -> dict:
        """Extract parameters from a qudi dat file.

        Args:
            filepath: Path to the qudi .dat file

        Returns:
            Dictionary of parameters
        """
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
                                params[label] = ast.literal_eval(
                                    inspect.cleandoc(value))
                        elif line.count(":") == 3:
                            # Handle files with timestamps in them
                            label = line.split(":")[0]
                            timestamp_str = "".join(line.split(":")[1:]).strip()
                            datetime_str = datetime.datetime.strptime(
                                timestamp_str,
                                "%d.%m.%Y %Hh%Mmin%Ss"
                            ).replace(tzinfo=datetime.timezone.utc)
                            params[label] = datetime_str
                    except Exception as _:
                        pass
        return params

    @_add_base_read_path
    @_check_extension(".dat")
    def read_into_dataframe(self, filepath: Path) -> pd.DataFrame:
        """Read a qudi data file into a pandas DataFrame for analysis.

        Args:
            filepath: Path to the qudi data file

        Returns:
            DataFrame containing the data from the qudi data file
        """
        with open(filepath) as handle:
            # Generate column names for DataFrame by parsing the file
            *_comments, names = itertools.takewhile(lambda line: line.startswith('#'),
                                                    handle)
            names = names[1:].strip().split("\t")
        return pd.read_csv(filepath, names=names, comment="#", sep="\t")

    @_add_base_read_path
    def read_csv(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """ Read a csv file into a pandas DataFrame. """
        return pd.read_csv(filepath, **kwargs)

    @_add_base_read_path
    def read_excel(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """ Read a csv file into a pandas DataFrame. """
        return pd.read_excel(filepath, **kwargs)

    @_add_base_read_path
    @_check_extension(".dat")
    def read_confocal_into_dataframe(self, filepath: Path) -> pd.DataFrame:
        """ Read a qudi confocal data file into a pandas DataFrame for analysis. """
        confocal_params = self.read_qudi_parameters(filepath)
        data = self.read_into_ndarray(filepath, delimiter="\t")
        # Use the confocal parameters to generate the index & columns for the DataFrame
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

    @_add_base_read_path
    def read_into_ndarray(self, filepath: Path, **kwargs) -> np.ndarray:
        """ Read a file into a numpy ndarray. """
        return np.genfromtxt(filepath, **kwargs)

    @_add_base_read_path
    def read_into_ndarray_transposed(self, filepath: Path, **kwargs) -> np.ndarray:
        """ Read a file into a transposed numpy ndarray. """
        return np.genfromtxt(filepath, **kwargs).T

    @_add_base_read_path
    @_check_extension(".pys")
    def read_pys(self, filepath: Path) -> dict:
        """ Read raw .pys data files into a dictionary. """
        byte_dict = np.load(str(filepath), encoding="bytes", allow_pickle=True)
        # Convert byte string keys to normal strings
        return {key.decode('utf8'): byte_dict.get(key) for key in byte_dict}

    @_add_base_read_path
    @_check_extension(".pkl")
    def read_pkl(self, filepath: Path) -> dict:
        """ Read pickle files into a dictionary. """
        with open(filepath, 'rb') as f:
            file = pickle.load(f)
        return file

    @_add_base_read_path
    @_check_extension(".dat")
    def read_nanonis_data(self, filepath: Path) -> pd.DataFrame:
        """Read data from a Nanonis .dat file.

        Args:
            filepath: Path to the Nanonis .dat file.

        Returns:
            DataFrame of data.
        """
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

    @_add_base_read_path
    @_check_extension(".dat")
    def read_nanonis_parameters(self, filepath: Path) -> dict:
        """Read parameters from a Nanonis .dat file.

        Args:
            filepath: Path to the Nanonis .dat file.

        Returns:
            Dictionary of parameters.
        """
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
                    with contextlib.suppress(ValueError):
                        value = float(value)
                    if "Oscillation Control>" in label:
                        label = label.replace("Oscillation Control>", "")
                    parameters[label] = value
        return parameters

    @_add_base_read_path
    @_check_extension(".sxm")
    def read_nanonis_spm_data(self, filepath: Path) -> pySPM.SXM:
        """Read a Nanonis .sxm data file.

        Args:
            filepath: Path to the .sxm file.

        Returns:
            pySPM.SXM object containing the data.
        """
        return pySPM.SXM(filepath)

    @_add_base_read_path
    @_check_extension(".001")
    def read_bruker_spm_data(self, filepath: Path) -> pySPM.Bruker:
        """Read a Bruker SPM data file.

        Args:
            filepath: Path to the .001 file.

        Returns:
            pySPM.Bruker object containing the data.
        """
        return pySPM.Bruker(filepath)

    @_add_base_read_path
    @_check_extension(".txt")
    def read_pfeiffer_data(self, filepath: Path) -> pd.DataFrame:
        """Read data stored by Pfeiffer vacuum monitoring software.

        Args:
            filepath: Path to the text file.

        Returns:
            DataFrame containing the data.
        """
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

    @_add_base_read_path
    @_check_extension(".xls")
    def read_lakeshore_data(self, filepath: Path) -> pd.DataFrame:
        """Read data stored by Lakeshore temperature monitor software.

        Args:
            filepath: Path to the Excel file.

        Returns:
            DataFrame containing the data.
        """
        # Extract only the origin timestamp
        origin = pd.read_excel(
            filepath, skiprows=1, nrows=1, usecols=[1], header=None
        )[1][0]
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

    @_add_base_read_path
    @_check_extension(".txt")
    def read_oceanoptics_data(self, filepath: str) -> pd.DataFrame:
        """Read spectrometer data from OceanOptics spectrometer.

        Args:
            filepath: Path to the data file.

        Returns:
            DataFrame containing the wavelength and intensity data.
        """
        df = pd.read_csv(filepath, sep="\t", skiprows=14,
                         names=["wavelength", "intensity"])
        return df

    @staticmethod
    def __get_forward_backward_counts(
            count_rates: np.ndarray,
            num_pixels: int
    ) -> tuple[np.ndarray, np.ndarray]:
        split_array = np.split(count_rates, 2 * num_pixels)
        # Extract forward scan array as every second element
        forward_counts = np.stack(split_array[::2])
        # Extract backward scan array as every shifted second element
        # Flip scan so that backward and forward scans represent the same data
        backward_counts = np.flip(np.stack(split_array[1::2]), axis=1)
        return forward_counts, backward_counts

    def read_pixelscanner_data(self, filepath: Path) -> (
            pySPM.SPM_image, pySPM.SPM_image):
        """ Read data from a PixelScanner measurement.

        Args:
            filepath: Path to the data file.

        Returns:
            Forward and backward scan data.
        """
        df = self.read_into_dataframe(filepath)
        num_pixels = int(np.sqrt(len(df) // 2))

        if num_pixels ** 2 != len(df) // 2:
            raise ValueError("Number of pixels does not match data length.")

        try:
            fwd, bwd = self.__get_forward_backward_counts(
                df["count_rates"].to_numpy(), num_pixels
            )
        except KeyError:
            try:
                fwd, bwd = self.__get_forward_backward_counts(
                    df["Count Rates (cps)"].to_numpy(), num_pixels
                )
            except KeyError:
                # Support old data format
                fwd = df["forward (cps)"].to_numpy().reshape(num_pixels, num_pixels)
                bwd = df["backward (cps)"].to_numpy().reshape(num_pixels, num_pixels)

        fwd = pySPM.SPM_image(fwd, channel="Forward", _type="NV-PL")
        bwd = pySPM.SPM_image(bwd, channel="Backward", _type="NV-PL")
        return fwd, bwd

    @_add_base_write_path
    @_check_extension(".pkl")
    def save_pkl(self, filepath: Path, obj: object):
        """Saves pickle files.

        Args:
            filepath: Path to the data file.
            obj: Object to be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    @_add_base_write_path
    @_check_extension(".pys")
    def save_pys(self, filepath: Path, dictionary: dict):
        """Saves .pys files.

        Args:
            filepath: Path to the data file.
            dictionary: Dictionary to be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(dictionary, f, 1)

    @_add_base_write_path
    @_check_extension(".pys")
    def save_df(self, filepath: Path, df: pd.DataFrame):
        """ Save Dataframe as csv. """
        df.to_csv(filepath, sep='\t', encoding='utf-8')

    @_add_base_write_path
    def save_figures(self, filepath: Path, fig: plt.Figure, **kwargs):
        """Saves figures from matplotlib plot data.

        By default, saves as jpg, png, pdf and svg.

        Args:
            fig: Matplotlib figure to save.
            filepath: Name of figure to save.
            only_jpg: If True, only save as jpg (default: False).
            only_pdf: If True, only save as pdf (default: False).
            **kwargs: Keyword arguments passed to matplotlib.pyplot.savefig().
        """
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
