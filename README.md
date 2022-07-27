# Qudi Hira Analysis

This toolkit automates a large portion of the work surrounding data analysis on quantum sensing experiments where the
primary raw data extracted is photon counts.

The high level interface is abstracted, and provides a set of functions to automate data import, handling and analysis.
It is designed to be exposed through Jupyter Notebooks, although the abstract interface allows it to be integrated into
larger, more general frameworks as well (with only some pain). Using the toolkit itself should only require a
beginner-level understanding of Python.

It also aims to improve transparency and reproducibility in experimental data analysis. In an ideal scenario,
two lines of code are sufficient to recreate all output data.

Python offers some very handy features like dataclasses, which are heavily used by this toolkit. Dataclasses offer a
full OOP (object-oriented programming) experience while analyzing complex data sets. They provide a solid and
transparent structure to the data to
reduce errors arising from data fragmentation. This generally comes at a large performance cost, but this is (largely)
sidestepped by lazy loading data and storing metadata instead wherever possible.

The visual structure of the toolkit is shown in the schema below. It largely consists of three portions:

- `IOHandler` assumes a central store of raw data, which is never modified (read-only)
- `DataHandler` automates the extraction of large amounts of data from the `IOHandler` interface
- `AnalysisLogic` contains a set of automated fitting routines using `lmfit` internally (built on top of fitting
  routines from the [qudi](https://github.com/Ulm-IQO/qudi) project)

This license of this project is located in the top level folder under `LICENSE`. Some specific files contain their
individual licenses in the file header docstring.

## Schema

### Overall

```mermaid
flowchart TD;
    IOHandler<-- Handle file paths, VPN and storage read/write operations -->PathHandler;
    PathHandler<-- Automated measurement data extraction and data handling -->DataHandler;
    Parameters-- Custom params for filepath handling -->PathHandler
    DataHandler-- Structure extracted data -->MeasurementDataclass;
    AnalysisLogic<-- Fit data -->MeasurementDataclass;
    MeasurementDataclass-- Plot fitted data --> Plot[Visualize data and add context in JupyterLab];
    Plot-- Save plotted data --> DataHandler;
    style MeasurementDataclass fill:#bbf,stroke:#f66,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
```

### Measurement Dataclass

```mermaid
flowchart TD;
    subgraph Standard Data
        MeasurementDataclass-->filepath1;
        MeasurementDataclass-->CustomDataImporter[Handle custom data like confocal etc.];
        CustomDataImporter-->data1;
        MeasurementDataclass-->params1;
        MeasurementDataclass-->timestamp1;
        MeasurementDataclass-->fit_result1;
        filepath1-->ParameterExtraction[Extract params from filename]
    end
    subgraph Pulsed Data
        MeasurementDataclass-- pulsed --oPulsedMeasurementDataclass;
        PulsedMeasurementDataclass-- measurement --oPulsedMeasurement;
        PulsedMeasurement--> filepath2;
        PulsedMeasurement--> data2;
        PulsedMeasurement--> params2;
        PulsedMeasurementDataclass-- laser_pulses --oLaserPulses; 
        LaserPulses--> filepath3;
        LaserPulses--> data3;
        LaserPulses--> params3;
        PulsedMeasurementDataclass-- timetrace --oRawTimetrace;
        RawTimetrace--> filepath4;
        RawTimetrace--> data4;
        RawTimetrace--> params4;
    end
```

### AnalysisLogic fits

| Dimension | Fit                           |
|-----------|-------------------------------|
| 1d        | decayexponential              |
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
| 2d        | twoDgaussian                  |

## Example: Plot all rabi oscillations in a timeframe and fit to exponentially decaying double sinusoid

```python
from dateutil.parser import parse
import matplotlib.pyplot as plt

from src.data_handler import DataHandler
from src.analysis_logic import FitMethods

tip_2S6 = DataHandler(measurement_folder="20220621_FR0612-F2-2S6_uhv")

tip_2S6_rabi_list = tip_2S6.load_measurements_into_dataclass_list(measurement_str="Rabi")
filtered_rabi_list = [rabi for rabi in tip_2S6_rabi_list if
                      parse("10 July 2022 13:30") < rabi.timestamp < parse("10 July 2022 17:30")]

fig, ax = plt.subplots(nrows=len(filtered_rabi_list))

for idx, rabi in enumerate(filtered_rabi_list):
    x, y, yerr = rabi.data["Controlled variable(s)"], rabi.data["Signal"], rabi.data["Error"]
    fit_x, fit_y, result = rabi.analysis.perform_fit(x=x, y=y, fit_function=FitMethods.sinedoublewithexpdecay)

    ax[idx].errorbar(x, y, yerr=yerr, fmt=".")
    ax[idx].plot(fit_x, fit_y, "-")
    ax[idx].set_title(f"Power = {rabi.get_param_from_filename(unit='dBm')}, "
                      f"T1rho = {result.params['Lifetime'].value}")

tip_2S6.save_figures(fig, filename="compare_rabi_oscillations_at different_powers")
```

For more examples see [ExampleNotebook.ipynb](ExampleNotebook.ipynb)

## Getting Started

### Prerequisites

Latest version of:

- [conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- [git](https://git-scm.com/downloads) version control system

### Clone the repository

#### With Git

```shell
git clone https://github.com/dineshpinto/qudi-hira-analysis.git
```

#### With Github CLI

```shell
gh repo clone dineshpinto/qudi-hira-analysis
```

### Installing dependencies

#### Creating the conda environment

```shell
conda env create -f tools/conda-env-xx.yml
```

where `xx` is either `win10`, `osx-intel` or `osx-apple-silicon`.

#### Activate environment

```shell
conda activate qudi-hira-analysis
```

#### Add conda environment to Jupyter kernel

```shell
python -m ipykernel install --user --name=qudi-hira-analysis
```

### Set up filepath parameters

Rename `parameters-example.py` to `parameters.py` and add in the correct data source and outputs. This will allow the library to automatically detect
which filepaths to use when connected remotely.

| Variable               | Explanation                                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------------|
| `lab_computer_name`    | Name of lab computer, use `os.environ["COMPUTERNAME"]` (eg. PCKK022)                                             |
| `remote_datafolder`    | Folder to connect to when running analysis remotely (eg. over VPN) (default: `\\kernix\qudiamond\Data`)          |
| `remote_output_folder` | Folder to place output images when running remotely (eg. over VPN) (default: `$USER\Documents\QudiHiraAnalysis`) |
| `local_datafolder`     | Folder to connect to when running  locally (default: `Z:\Data`)                                                  |
| `local_output_folder`  | Folder to place output images when running locally (default: `Z:\QudiHiraAnalysis`)                              |

### Start the analysis

```shell
jupyter lab
```

## Makefile

The Makefile is configured to generate a variety of outputs:

+ `make pdf` : Converts all notebooks to PDF (requires LaTeX backend)
+ `make html`: Converts all notebooks to HTML files
+ `make py`  : Converts all notebooks to Python files (useful for VCS)
+ `make all` : Sequentially runs all the notebooks in folder

To use the `make` command on Windows you can install [Chocolatey](https://chocolatey.org/install), then
install make with `choco install make`
