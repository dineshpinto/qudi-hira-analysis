# -*- coding: utf-8 -*-
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

Copyright (c) 2020 Dinesh Pinto. See the LICENSE file at the
top-level directory of this distribution and at <https://github.com/dineshpinto/qudiamond-analysis/>
"""

import datetime
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser, tz

"""
Functions for reading and writing into data files.
"""


def load_pys(filename, folder=""):
    """ Loads raw pys data files. Wraps around numpy.load. """
    path = "../raw_data/" + folder
    if filename.endswith('.pys'):
        return np.load(path + filename, encoding="bytes", allow_pickle=True)
    else:
        return np.load(path + filename + ".pys", encoding="bytes", allow_pickle=True)


def save_pys(dictionary, filename, folder=""):
    """ Saves processed pickle files for plotting/further analysis. """
    path = "../data/" + folder
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + filename + '.pys', 'wb') as f:
        pickle.dump(dictionary, f, 1)


def save_df(df, filename, folder=""):
    """ Save Dataframe as csv. """
    path = "../data/" + folder
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + filename + ".csv", sep='\t', encoding='utf-8')


def load_pkl(filename, folder=""):
    """ Loads processed pickle files for plotting/further analysis. """
    path = "../data/" + folder
    with open(path + filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, filename, folder=""):
    """ Saves processed pickle files for plotting/further analysis. """
    path = "../data/" + folder
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def save_figures(filename, folder="", overwrite=True):
    """ Saves figures from matplotlib plot data. """
    path = "../figures/" + folder
    if not os.path.exists(path):
        os.makedirs(path)

    exts = [".pdf", ".svg", ".png"]

    if filename.endswith(".pys"):
        filename, _ = os.path.splitext(filename)

    if not overwrite:
        for ext in exts:
            if os.path.isfile(path + filename + ext):
                raise IOError(filename + ext + " already exists.")

    for ext in exts:
        if ext == ".png":
            dpi = 600
        else:
            dpi = 1000
        plt.savefig(path + filename + ext, dpi=dpi, bbox_inches="tight",
                    metadata={"Title": "{}".format(filename), "Author": "Dinesh Pinto"})


"""
Functions for reading Nanonis data files
"""


def extract_data_from_dat(filename, folder=""):
    """ Extract data from a Nanonis dat file. """
    if not filename.endswith(".dat"):
        filename += ".dat"

    with open(folder + filename) as dat_file:
        for num, line in enumerate(dat_file, 1):
            if "[DATA]" in line:
                # Find number of rows to skip when extracting data
                skiprows = num
                break

    df = pd.read_table(folder + filename, sep="\t", skiprows=skiprows)
    return df


def extract_parameters_from_dat(filename, folder=""):
    """ Extract parameters from a Nanonis dat file. """
    if not filename.endswith(".dat"):
        filename += ".dat"

    d = {}
    with open(folder + filename) as dat_file:
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
                d[label] = value
    return d


def read_dat(filename, folder=""):
    """
    Convenience function to extract both parameters and data from a DAT file.

    Args:
        filename: string
            Name of DAT filename stored on disk
        folder:
            Folderpath on disk on disk


    Returns:
        parameters : dict
            header parameters extracted from DAT file
        data : dict
            data columns extracted from DAT file
    """
    parameters = extract_parameters_from_dat(filename, folder=folder)
    data = extract_data_from_dat(filename, folder=folder)
    return parameters, data


def get_folderpath(folder_name):
    """
    Create absolute folder paths.

    Args:
        folder_name: string
            Folder from the "Data" directory on computer

    Returns: string
        Full filepath of the directory depending on the PC name

    """
    if os.environ['COMPUTERNAME'] == 'NBKK055':
        return r"C:\\Nextcloud\\Data\\{}\\".format(folder_name)
    else:
        return r"Z:\\Data\\{}\\".format(folder_name)


def savefig(filename=None, folder=None, **kwargs):
    """
    General function to save figures. Wraps around matplotlib.pyplot.savefig().

    Args:
        filename: string
            name of file to save on disk
                without extension saves ".jpg"  and ".svg"
                with extension only saves specific extension
        folder: string
            name of folder location to save on disk. Creates sub-directory "figures/" if it does not exist.
        **kwargs: matplotlib.plot(**kwargs)
    """
    """  """
    if folder is None:
        folder = "../figures/"
    else:
        folder += r"figures\\"
        if not os.path.exists(folder):
            os.makedirs(folder)

    if filename is None:
        filename = "image"

    # Extract just the filename without extension
    fname, ext = os.path.splitext(filename)

    if not ext:
        # If no extension given, use some sane defaults
        extensions = [".jpg", ".svg"]
    else:
        extensions = [ext]

    for extension in extensions:
        try:
            if not kwargs:
                # Sane defaults for saving
                plt.savefig(folder + fname + extension, dpi=600, bbox_inches="tight")
            else:
                plt.savefig(folder + fname + extension, **kwargs)
        except AttributeError:
            # Happens when using JupyterLab with ipympl, can be safely ignored
            pass


def channel_to_gauge_names(channel_names):
    """ Replace the channel names with gauge locations. """
    gauges = {"CH 1": "Main", "CH 2": "Prep", "CH 3": "Backing"}
    return [gauges.get(ch, ch) for ch in channel_names]


def convert_tpg_to_mpl_time(df):
    """ Read DataFrame extracted using read_tpg_data and add in matplotlib datetimes using "Date" and "Time" cols. """
    datetimes = df["Date"] + " " + df["Time"]
    # Convert raw dates and times to datetime Series, then to an matplotlib Series
    dt_series_datetime = [datetime.datetime.strptime(str(dt), '%d-%b-%y %H:%M:%S.%f') for dt in datetimes]
    dt_series_mpl = matplotlib.dates.date2num(dt_series_datetime)
    return dt_series_mpl


def read_tpg_data(filename, folder=None):
    """
     Read data stored by Pfeiffer vacuum monitoring software.

    Args:
        filename: string
            name of ".txt" file on disk
        folder: string
            location of file on disk

    Returns:
        df : pandas.DataFrame
            DataFrame with all .txt columns and converted matplotlib timestamps

    """
    if not filename.endswith(".txt"):
        filename += ".txt"

    # Extract only the header to check which gauges are connected
    file_header = pd.read_csv(folder + filename, sep="\t", skiprows=1, nrows=1)
    header = channel_to_gauge_names(file_header)

    # Create DataFrame with new header
    df = pd.read_csv(folder + filename, sep="\t", skiprows=5, names=header)

    # Save matplotlib datetimes for plotting
    df["MPL_datetimes"] = convert_tpg_to_mpl_time(df)
    return df


def read_tm224_data(filename, folder=None):
    """
     Read data stored by Lakeshore TM224 temperature monitor software.

    Args:
        filename: string
            name of ".xls" file on disk
        folder: string
            location of file on disk

    Returns:
        df : pandas.DataFrame
            DataFrame with all .xls columns and converted matplotlib timestamps
    """
    if not filename.endswith(".xls"):
        filename += ".xls"

    # Extract only the timestamp
    timestamp = pd.read_excel(folder + filename, skiprows=1, nrows=1, usecols=[1], header=None)[1][0]
    # Parse datetime object from timestamp
    timestamp_dt = parser.parse(timestamp, tzinfos={"CET": tz.gettz('Europe/Berlin')})

    # Create DataFrame
    df = pd.read_excel(folder + filename, skiprows=3)

    # Add matplotlib datetimes to DataFrame
    time_array = []
    for milliseconds in df["Time"]:
        time_array.append(timestamp_dt + datetime.timedelta(milliseconds=milliseconds))
    df["MPL_datetimes"] = matplotlib.dates.date2num(time_array)

    return df
