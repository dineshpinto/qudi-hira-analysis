from __future__ import annotations

import logging
import os
from typing import List

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)


class PathHandler:
    def __init__(self, data_folder: str, figure_folder: str, measurement_folder: str):
        self.log = logging.getLogger(__name__)

        self.data_folder_path = self.__get_data_folder_path(data_folder, measurement_folder)
        self.figure_folder_path = self.__get_figure_folder_path(figure_folder, measurement_folder)

    def __get_data_folder_path(self, data_folder: str, folder_name: str) -> str:
        """ Check if folder exists and return absolute folder paths. """
        path = os.path.join(data_folder, folder_name)

        if not os.path.exists(path):
            raise IOError("Data folder path does not exist.")

        self.log.info(f"Data folder path is {path}")
        return path

    def __get_figure_folder_path(self, figure_folder: str, folder_name: str) -> str:
        """ Check if folder exists, if not, create it and return absolute folder paths. """

        if not os.path.exists(figure_folder):
            self.log.info(f"Creating new output folder {figure_folder}")
            os.mkdir(figure_folder)

        path = os.path.join(figure_folder, folder_name)

        if not os.path.exists(path):
            self.log.info(f"Creating new output folder path {path}")
            os.makedirs(path)
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
