from __future__ import annotations

import logging
import os
from typing import List

import parameters as params

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class PathHandler:
    def __init__(self, measurement_folder: str):
        self.log = logging.getLogger(__name__)
        self.data_folder_path = self.__get_data_folder_path(measurement_folder)
        self.figure_folder_path = self.__get_figure_folder_path(measurement_folder)

    def __get_data_folder_path(self, folder_name: str) -> str:
        """ Create absolute folder paths. """
        if os.environ["COMPUTERNAME"] == params.lab_computer_name:
            path = os.path.join(params.local_datafolder, folder_name)
        else:
            path = os.path.join(params.remote_datafolder, folder_name)

        self.log.info(f"Data folder path is {path}")
        return path

    def __get_figure_folder_path(self, folder_name: str) -> str:
        if os.environ["COMPUTERNAME"] == params.lab_computer_name:
            path = os.path.join(params.local_output_folder, folder_name)
        else:
            path = os.path.join(params.remote_output_folder, folder_name)

        if not os.path.exists(path):
            self.log.info(f"Creating new figure folder path {path}")
            os.mkdir(path)
        else:
            self.log.info(f"Figure folder path is {path}")
        return path

    def get_measurement_filepaths(self, measurement: str, extension: str = ".dat",
                                  exclude_str: str = "image_1.dat") -> list:
        """
        List all measurement files for a single measurement type, regardless of date
        within a similar set (i.e. top level folder).
        """
        filepaths: List[str] = []
        for root, dirs, files in os.walk(self.data_folder_path):
            for file in files:
                # Check if measurement string is in the root of the folder walk
                if (measurement in root or measurement in file) and (exclude_str not in file):
                    if extension:
                        if file.endswith(extension):
                            filepaths.append(os.path.join(root, file))
                    else:
                        filepaths.append(os.path.join(root, file))
        return filepaths
