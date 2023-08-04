"""Analytics suite for qubit SPM using FPGA timetaggers

## Getting started

```python
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from qudi_hira_analysis import DataHandler

dh = DataHandler(
    data_folder=Path("C:\\Data"), # Path to the data folder
    figure_folder=Path("C:\\QudiHiraAnalysis"), # Path to the figure folder
    measurement_folder=Path("20230101_NV1") # Name of the measurement folder
 )

# Search and lazy-load files with "odmr" in the name
odmr_measurements = dh.load_measurements("odmr")
```

To load a specific set of measurements from the data folder, use the `dh.load_measurements()` method, which takes the
following required arguments:

- `measurement_str` is the string that is used to identify the measurement. It is used to filter the data files in the
  `data_folder` and `measurement_folder` (if specified)

Optional arguments:

- `qudi` is a boolean. If `True`, the data is assumed to be in the format used by Qudi (default: True)
- `pulsed` is a boolean. If `True`, the data is assumed to be in the format used by Qudi for pulsed measurements
(default: False)
- `extension` is the extension of the data files (default: ".dat")

The `load_measurements` function returns a dictionary containing the measurement data filtered by `measurement_str`.
The dictionary keys are measurement timestamps in "(year)(month)(day)-(hour)(minute)-(second)" format.


## Examples

### NV-ODMR map
```python
# Extract ODMR measurements from the measurement folder
odmr_measurements = dh.load_measurements(measurement_str="2d_odmr_map")
odmr_measurements = dict(sorted(odmr_measurements.items()))

# Perform parallel (=num CPU cores) ODMR fitting
odmr_measurements = dh.fit_raster_odmr(odmr_measurements)

# Calculate 2D ODMR map from the fitted ODMR measurements
pixels = int(np.sqrt(len(odmr_measurements)))
image = np.zeros((pixels, pixels))

for idx, odmr in enumerate(odmr_measurements.values()):
  row, col = odmr.xy_position
  if len(odmr.fit_model.params) > 6:
    # Calculate double Lorentzian splitting
    image[row, col] = np.abs(odmr.fit_model.best_values["l1_center"]
                             - odmr.fit_model.best_values["l0_center"])


map = sns.heatmap(image, cbar_kws={"label": r"$\Delta E$ (MHz)"})

# Save the figure to the figure folder specified earlier
dh.save_figures(filepath="2d_odmr_map_with_residuals", fig=map.get_figure(), only_jpg=True)
```


### NV-PL map

```python
pixel_scanner_measurements = dh.load_measurements(measurement_str="PixelScanner")

fwd, bwd = pixel_scanner_measurements["20230101-0420-00"].data

# If size is known, it can be specified here
fwd.size["real"] = {"x": 1e-6, "y": 1e-6, "unit": "m"}

fig, ax = plt.subplots()

# Perform (optional) image corrections
fwd.filter_gaussian(sigma=0.5)

# Add scale bar, color bar and plot the data
img = fwd.show(cmap="inferno", ax=ax)
fwd.add_scale(length=1e-6, ax=ax, height=1)
cbar = fig.colorbar(img)
cbar.set_label("NV-PL (kcps)")

# Save the figure to the figure folder specified earlier
dh.save_figures(filepath="nv_pl_scan", fig=fig, only_jpg=True)
```

### Nanonis AFM measurements

```python
afm_measurements = dh.load_measurements(measurement_str="Scan", extension=".sxm", qudi=False)

afm = afm_measurements["20230101-0420-00"].data

# Print the channels available in the data
afm.list_channels()
topo = afm.get_channel("Z")

fig, ax = plt.subplots()

# Perform (optional) image corrections
topo.correct_lines()
topo.correct_plane()
topo.filter_lowpass(fft_radius=20)
topo.zero_min()

# Add scale bar, color bar and plot the data
img = topo.show(cmap="inferno", ax=ax)
topo.add_scale(length=1e-6, ax=ax, height=1, fontsize=10)
cbar = fig.colorbar(img)
cbar.set_label("Height (nm)")

dh.save_figures(filepath="afm_topo", fig=fig, only_jpg=True)
```

### g(2) measurements (anti-bunching fit)

```python
autocorrelation_measurements = dh.load_measurements(measurement_str="Autocorrelation")

fig, ax = plt.subplots()

for autocorrelation in autocorrelation_measurements.values():
    autocorrelation.data["Time (ns)"] = autocorrelation.data["Time (ps)"] * 1e-3
    # Plot the data
    sns.lineplot(data=autocorrelation.data, x="Time (ns)", y="g2(t) norm", ax=ax)
    # Fit the data using the antibunching function
    fit_x, fit_y, result = dh.fit(x="Time (ns)", y="g2(t) norm", data=autocorrelation.data,
                                  fit_function=dh.fit_function.antibunching)
    # Plot the fit
    sns.lineplot(x=fit_x, y=fit_y, ax=ax, color="C1")

# Save the figure to the figure folder specified earlier
dh.save_figures(filepath="autocorrelation_variation", fig=fig)
```

### ODMR measurements (double Lorentzian fit)

```python
odmr_measurements = dh.load_measurements(measurement_str="ODMR", pulsed=True)

fig, ax = plt.subplots()

for odmr in odmr_measurements.values():
    sns.scatterplot(data=odmr.data, x="Controlled variable(Hz)", y="Signal", ax=ax)
    fit_x, fit_y, result = dh.fit(x="Controlled variable(Hz)", y="Signal", data=odmr.data,
                                  fit_function=dh.fit_function.lorentziandouble)
    sns.lineplot(x=fit_x, y=fit_y, ax=ax, color="C1")

dh.save_figures(filepath="odmr_variation", fig=fig)
```

### Rabi measurements (sine exp. decay fit)

```python
rabi_measurements = dh.load_measurements(measurement_str="Rabi", pulsed=True)

fig, ax = plt.subplots()

for rabi in rabi_measurements.values():
    sns.scatterplot(data=rabi.data, x="Controlled variable(s)", y="Signal", ax=ax)
    fit_x, fit_y, result = dh.fit(x="Controlled variable(s)", y="Signal", data=rabi.data,
                                  fit_function=dh.fit_function.sineexponentialdecay)
    sns.lineplot(x=fit_x, y=fit_y, ax=ax, color="C1")

dh.save_figures(filepath="rabi_variation", fig=fig)
```

### Temperature measurements

```python
temperature_measurements = dh.load_measurements(measurement_str="Temperature", qudi=False)

temperature = pd.concat([t.data for t in temperature_measurements.values()])

fig, ax = plt.subplots()
sns.lineplot(data=temperature, x="Time", y="Temperature", ax=ax)
dh.save_figures(filepath="temperature_monitoring", fig=fig)
```

### Bruker MFM measurements

```python
bruker_measurements = dh.load_measurements(measurement_str="", extension=".001", qudi=False)

bruker_data = bruker_measurements["20230101-0420-00"].data

# Print the channels available in the data
bruker_data.list_channels()
mfm = bruker_data.get_channel("Phase", mfm=True)

fig, ax = plt.subplots()

# Perform (optional) image corrections
mfm.correct_plane()
mfm.zero_min()

# Add scale bar, color bar and plot the data
img = mfm.show(cmap="inferno", ax=ax)
mfm.add_scale(length=1, ax=ax, height=1, fontsize=10)
cbar = fig.colorbar(img)
cbar.set_label("MFM contrast (deg)")

dh.save_figures(filepath="MFM", fig=fig, only_jpg=True)
```

### PYS data (pi3diamond compatibility)

```python
pys_measurements = dh.load_measurements(measurement_str="ndmin", extension=".pys", qudi=False)
pys = pys_measurements[list(pys_measurements)[0]].data

fig, ax = plt.subplots()
sns.lineplot(x=pys["time_bins"], y=pys["counts"], ax=ax)
dh.save_figures(filepath="pys_measurement", fig=fig)
```

"""

from .analysis_logic import AnalysisLogic
from .data_handler import DataHandler
from .io_handler import IOHandler

__all__ = ["DataHandler", "IOHandler", "AnalysisLogic"]
