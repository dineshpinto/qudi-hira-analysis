import os
from dataclasses import dataclass

from abc import ABC, abstractmethod


@dataclass(frozen=True)
class Parameters(ABC):
    @property
    @abstractmethod
    def lab_computer_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def remote_datafolder(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def remote_output_folder(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def local_datafolder(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def local_output_folder(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def assertion_check(self):
        raise NotImplementedError


@dataclass(frozen=True)
class QudiHiraParameters(Parameters):
    """ Parameters for where the data ís stored and where it should be saved """
    # Name of lab computer (run `os.environ["COMPUTERNAME"]` to check)
    lab_computer_name: str = "PCKK022"

    # The code automatically detects whether kernix is connected remotely or not

    # Data folder on kernix when connected remotely (eg. VPN)
    remote_datafolder: str = os.path.join("\\\\kernix", "qudiamond", "Data")
    # Folder to save output images
    remote_output_folder: str = os.path.join(os.path.expanduser("~"), "QudiHiraAnalysis")

    # Data folder on kernix when connected directly (e.g. on lab PC)
    local_datafolder: str = os.path.join("Z:/", "Data")
    # Folder to save output images
    local_output_folder: str = os.path.join("Z:/", "QudiHiraAnalysis")

    def __post_init__(self):
        self.assertion_check()

    def assertion_check(self):
        assert self.remote_datafolder != self.remote_output_folder, "Input and output folders must be distinct"
        assert self.local_datafolder != self.local_output_folder, "Input and output folders must be distinct"


@dataclass(frozen=True)
class DiamondAFMParameters(Parameters):
    """ Parameters for where the data ís stored and where it should be saved """
    # Name of lab computer (run `os.environ["COMPUTERNAME"]` to check)
    lab_computer_name: str = "LABKK022"

    # The code automatically detects whether kernix is connected remotely or not

    # Data folder on kernix when connected remotely (eg. VPN)
    remote_datafolder: str = os.path.join("\\\\kernix", "diamond_AFM", "data")
    # Folder to save output images
    remote_output_folder: str = os.path.join(os.path.expanduser("~"), "QudiHiraAnalysis")

    # Data folder on kernix when connected directly (e.g. on lab PC)
    local_datafolder: str = os.path.join("Z:/", "data")
    # Folder to save output images
    local_output_folder: str = os.path.join("Z:/", "QudiHiraAnalysis")

    def __post_init__(self):
        self.assertion_check()

    def assertion_check(self):
        assert self.remote_datafolder != self.remote_output_folder, "Input and output folders must be distinct"
        assert self.local_datafolder != self.local_output_folder, "Input and output folders must be distinct"
