# Rename this file to parameters.py after modifying it
import os

# Name of lab computer (run `os.environ["COMPUTERNAME"]` on lab computer to check)
lab_computer_name: str = "PCKK022"

# REMOTE FOLDER PATHS (e.g. when connected over VPN)

# Source data folder on kernix
remote_datafolder: str = os.path.join("\\\\kernix", "qudiamond", "Data")
# Folder to save output images
remote_output_folder: str = os.path.join(os.path.expanduser("~"), "QudiHiraAnalysis")

# LOCAL FOLDER PATHS (e.g. when on lab PC)

# Source data folder on kernix
local_datafolder: str = os.path.join("Z:/", "Data")
# Folder to save output images
local_output_folder: str = os.path.join("Z:/", "QudiHiraAnalysis")
