import importlib.metadata

try:
    __version__ = importlib.metadata.version("qudi_hira_analysis")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
