import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    path = "../../Data/" + folder
    if not filename.endswith(".dat"):
        filename += ".dat"

    with open(path + filename) as dat_file:
        for num, line in enumerate(dat_file, 1):
            if "[DATA]" in line:
                # Find number of rows to skip when extracting data
                skiprows = num
                break

    df = pd.read_table(path + filename, sep="\t", skiprows=skiprows)
    return df


def extract_parameters_from_dat(filename, folder=""):
    path = "../../Data/" + folder
    if not filename.endswith(".dat"):
        filename += ".dat"

    d = {}
    with open(path + filename) as dat_file:
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
    parameters = extract_parameters_from_dat(filename, folder=folder)
    data = extract_data_from_dat(filename, folder=folder)
    return parameters, data
