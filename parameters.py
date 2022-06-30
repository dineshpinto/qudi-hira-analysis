import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ParametersQudiHira:
    """ Parameters for new system """
    # Name of lab computer (run `os.environ["COMPUTERNAME"]` to check)
    lab_computer_name: str = "PCKK022"

    # The code automatically detects whether kernix is connected remotely or not

    # Data folder on kernix when connected remotely (eg. VPN)
    kernix_remote_datafolder: str = os.path.join("\\\\kernix", "qudiamond", "Data")
    # Folder to save output images
    output_figure_remote_folder: str = os.path.join(os.path.expanduser("~"), "Data_Analysis")

    # Data folder on kernix when connected directly (e.g. on lab PC)
    kernix_local_datafolder: str = os.path.join("Z:/", "Data")
    # Folder to save output images
    output_figure_local_folder: str = os.path.join("Z:/", "Data_Analysis")


@dataclass(frozen=True)
class ParametersDiamondAFM:
    """ Parameters for old system """
    # Name of lab computer (run `os.environ["COMPUTERNAME"]` to check)
    lab_computer_name: str = "PCKK022"

    # The code automatically detects whether kernix is connected remotely or not

    # Data folder on kernix when connected remotely (eg. VPN)
    kernix_remote_datafolder: str = os.path.join("\\\\kernix", "diamond_AFM", "Data")
    # Folder to save output images
    output_figure_remote_folder: str = os.path.join(os.path.expanduser("~"), "Data_Analysis")

    # Data folder on kernix when connected directly (e.g. on lab PC)
    kernix_local_datafolder: str = os.path.join("Z:/", "data")
    # Folder to save output images
    output_figure_local_folder: str = os.path.join("Z:/", "Data_Analysis")
