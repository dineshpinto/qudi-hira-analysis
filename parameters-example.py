import os

# Source data folder
# ------------------
# This can be a remote (i.e. over VPN) or local folder
data_folder: str = os.path.join("C:/", "Data")

# Folder to save output images
# ----------------------------
# By default it will save it under the home directory in a folder called "QudiHiraAnalysis"
figure_folder: str = os.path.join(os.path.expanduser("~"), "QudiHiraAnalysis")
