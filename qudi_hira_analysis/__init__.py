"""Analytics suite for qubit SPM using FPGA timetaggers

## Getting started

```python
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from qudi_hira_analysis import DataHandler

dh = DataHandler(
    data_folder=Path("C:/Data"), # Path to the data folder
    figure_folder=Path("C:/QudiHiraAnalysis"), # Path to the figure folder
    measurement_folder=Path("20230101_NV1") # Name of the measurement folder
 )
```
Start by creating an instance of the `DataHandler` class. Specify the location you want to load data from
(`data_folder`), the location you want to save figures to (`figure_folder`) and (optionally) the
name of the measurement folder (`measurement_folder`). If a measurement folder is specified, its path will be combined
with the data folder path to form the full path to the measurement data.

### Loading data

```python
# Search and lazy-load files with "odmr" in the path
odmr_measurements = dh.load_measurements("odmr")
odmr = odmr_measurements["20230101-0420-00"]
```
To load a specific set of measurements from the data folder, use the `DataHandler.load_measurements()` method.
The method takes a string as an argument and searches for files with the string in the path. The files are lazy-loaded,
so the data is only loaded when it is needed. The method returns a dictionary, where the keys are the timestamps of the
measurements and the values are `measurement_dataclass.MeasurementDataclass()` objects.

### Fitting data

```python
x_fit, y_fit, result = dh.fit(x="Controlled variable(Hz)", y="Signal",
                              fit_function=dh.fit_function.doublelorentzian,
                              data=odmr.data)

# Plot the data and the fit
plot = sns.scatterplot(x="Freq", y="Counts", data=odmr.data, label=odmr.timestamp)
sns.lineplot(x=xf, y=yf, ax=plot, label="Fit")

# Generate fit report
print(res.fit_report())
```
To fit data, call the `DataHandler.fit()` method. This method accepts pandas DataFrames, numpy arrays or pandas Series
as inputs. To get the full list of available fit routines, explore the `DataHandler.fit_function` attribute or call
`AnalysisLogic.get_all_fits()`.
The fit functions are:

| Dimension | Fit                           |
|-----------|-------------------------------|
| 1D        | decayexponential              |
|           | biexponential                 |
|           | decayexponentialstretched     |
|           | gaussian                      |
|           | gaussiandouble                |
|           | gaussianlinearoffset          |
|           | hyperbolicsaturation          |
|           | linear                        |
|           | lorentzian                    |
|           | lorentziandouble              |
|           | lorentziantriple              |
|           | sine                          |
|           | sinedouble                    |
|           | sinedoublewithexpdecay        |
|           | sinedoublewithtwoexpdecay     |
|           | sineexponentialdecay          |
|           | sinestretchedexponentialdecay |
|           | sinetriple                    |
|           | sinetriplewithexpdecay        |
|           | sinetriplewiththreeexpdecay   |
| 2D        | twoDgaussian                  |


### Saving data

```python
# Save the figure to the figure folder specified earlier
dh.save_figures(filepath=Path("odmr"), fig=plot.get_figure(),
                only_pdf=True, bbox_inches="tight")
```

To save figures, call the `DataHandler.save_figures()` method. By default,
the figures are saved as JPG, PDF, PNG and SVG.
This can be changed by setting the `only_jpg` or `only_pdf` arguments to `True`.

## Examples

### NV-ODMR map
Extract a heatmap of ODMR splittings from a 2D raster NV-ODMR map.

```python
# Extract ODMR measurements from the measurement folder
odmr_measurements = dh.load_measurements("2d_odmr_map")
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


map = sns.heatmap(image, cbar_kws={"label": "Delta E (MHz)"})

# Save the figure to the figure folder specified earlier
dh.save_figures(filepath="2d_odmr_map", fig=map.get_figure(), only_jpg=True)
```


### NV-PL map
Extract a heatmap of NV photo-luminescence from a 2D raster NV-PL map.

```python
pixel_scanner_measurements = dh.load_measurements("PixelScanner")

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
Extract a heatmap of AFM data from a 2D raster Nanonis AFM scan.

```python
afm_measurements = dh.load_measurements("Scan", extension=".sxm", qudi=False)

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
Extract a anti-bunching fit from a g(2) measurement.

```python
autocorrelation_measurements = dh.load_measurements("Autocorrelation")

fig, ax = plt.subplots()

for autocorrelation in autocorrelation_measurements.values():
    autocorrelation.data["Time (ns)"] = autocorrelation.data["Time (ps)"] * 1e-3
    # Plot the data
    sns.lineplot(data=autocorrelation.data, x="Time (ns)", y="g2(t) norm", ax=ax)
    # Fit the data using the antibunching function
    fit_x, fit_y, result = dh.fit(x="Time (ns)", y="g2(t) norm",
                                  data=autocorrelation.data,
                                  fit_function=dh.fit_function.antibunching)
    # Plot the fit
    sns.lineplot(x=fit_x, y=fit_y, ax=ax, color="C1")

# Save the figure to the figure folder specified earlier
dh.save_figures(filepath="autocorrelation_variation", fig=fig)
```

### ODMR measurements (double Lorentzian fit)
Extract a double Lorentzian fit from an ODMR measurement.

```python
odmr_measurements = dh.load_measurements("ODMR", pulsed=True)

fig, ax = plt.subplots()

for odmr in odmr_measurements.values():
    sns.scatterplot(data=odmr.data, x="Controlled variable(Hz)", y="Signal", ax=ax)
    fit_x, fit_y, result = dh.fit(x="Controlled variable(Hz)", y="Signal",
                                  data=odmr.data,
                                  fit_function=dh.fit_function.lorentziandouble)
    sns.lineplot(x=fit_x, y=fit_y, ax=ax, color="C1")

dh.save_figures(filepath="odmr_variation", fig=fig)
```

### Rabi measurements (sine exp. decay fit)
Extract a exponentially decaying sine fit from a Rabi measurement.

```python
rabi_measurements = dh.load_measurements("Rabi", pulsed=True)

fig, ax = plt.subplots()

for rabi in rabi_measurements.values():
    sns.scatterplot(data=rabi.data, x="Controlled variable(s)", y="Signal", ax=ax)
    fit_x, fit_y, result = dh.fit(x="Controlled variable(s)", y="Signal",
                                  data=rabi.data,
                                  fit_function=dh.fit_function.sineexponentialdecay)
    sns.lineplot(x=fit_x, y=fit_y, ax=ax, color="C1")

dh.save_figures(filepath="rabi_variation", fig=fig)
```

### Temperature measurements
Extract temperature data from a Lakeshore temperature monitor.

```python
temperature_measurements = dh.load_measurements("Temperature", qudi=False)

temperature = pd.concat([t.data for t in temperature_measurements.values()])

fig, ax = plt.subplots()
sns.lineplot(data=temperature, x="Time", y="Temperature", ax=ax)
dh.save_figures(filepath="temperature_monitoring", fig=fig)
```

### Bruker MFM measurements
Extract a heatmap of MFM data from a 2D raster Bruker MFM map.

```python
bruker_measurements = dh.load_measurements("mfm", extension=".001", qudi=False)

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
pys_measurements = dh.load_measurements("ndmin", extension=".pys", qudi=False)
pys = pys_measurements[list(pys_measurements)[0]].data

fig, ax = plt.subplots()
sns.lineplot(x=pys["time_bins"], y=pys["counts"], ax=ax)
dh.save_figures(filepath="pys_measurement", fig=fig)
```

"""

from .analysis_logic import AnalysisLogic, FitMethodsAndEstimators
from .data_handler import DataHandler
from .io_handler import IOHandler

__all__ = ["DataHandler", "IOHandler", "AnalysisLogic", "FitMethodsAndEstimators"]
