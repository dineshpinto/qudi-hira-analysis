import os
from dataclasses import dataclass

from abc import ABC, abstractmethod


@dataclass(frozen=True)
class Parameters(ABC):
    @property
    @abstractmethod
    def lab_computer_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def kernix_remote_datafolder(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_figure_remote_folder(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def kernix_local_datafolder(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_figure_local_folder(self):
        raise NotImplementedError


@dataclass(frozen=True)
class QudiHiraParameters(Parameters):
    """ Parameters for new system """
    # Name of lab computer (run `os.environ["COMPUTERNAME"]` to check)
    lab_computer_name: str = "PCKK022"

    # The code automatically detects whether kernix is connected remotely or not

    # Data folder on kernix when connected remotely (eg. VPN)
    kernix_remote_datafolder: str = os.path.join("\\\\kernix", "qudiamond", "Data")
    # Folder to save output images
    output_figure_remote_folder: str = os.path.join(os.path.expanduser("~"), "QudiHiraAnalysis")

    # Data folder on kernix when connected directly (e.g. on lab PC)
    kernix_local_datafolder: str = os.path.join("Z:/", "Data")
    # Folder to save output images
    output_figure_local_folder: str = os.path.join("Z:/", "QudiHiraAnalysis")


@dataclass(frozen=True)
class DiamondAFMParameters:
    """ Parameters for old system """
    # Name of lab computer (run `os.environ["COMPUTERNAME"]` to check)
    lab_computer_name: str = "PCKK022"

    # The code automatically detects whether kernix is connected remotely or not

    # Data folder on kernix when connected remotely (eg. VPN)
    kernix_remote_datafolder: str = os.path.join("\\\\kernix", "diamond_AFM", "data")
    # Folder to save output images
    output_figure_remote_folder: str = os.path.join(os.path.expanduser("~"), "QudiHiraAnalysis")

    # Data folder on kernix when connected directly (e.g. on lab PC)
    kernix_local_datafolder: str = os.path.join("Z:/", "data")
    # Folder to save output images
    output_figure_local_folder: str = os.path.join("Z:/", "QudiHiraAnalysis")
