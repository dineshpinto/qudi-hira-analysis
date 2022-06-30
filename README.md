# Qudi Hira Analysis

A reproducible and transparent toolkit to analyze experimental data, primarily from an imaging magnetometer in diamond.

**Transparency** is achieved using Jupyter notebooks, which mix analysis code and figures along with written text and
equations. The
toolkit itself is built entirely on free and open-source software.

**Reproducibility** is achieved using automated build tools (GNU Make) and environment metadata storage. Two lines of
code are sufficient to reproduce all analyzed data and figures.

This license of this project is located in the top level folder under `LICENSE`. Some specific files contain their
individual licenses in the file header docstring.

## Schema

### Overall
```mermaid
flowchart TD;
    GenericIO<-- Handle file paths, VPN and storage read/write operations -->PathHandler;
    PathHandler<-- Automated measurement data extraction and data handling -->DataHandler;
    Parameters-- Custom params for filepath handling -->PathHandler
    DataHandler-- Structure extracted data -->MeasurementDataclass;
    MeasurementDataclass-- Fit and analyze data -->AnalysisLogic;
    AnalysisLogic-- Plot fitted data --> Plot[Visualize data and add context in JupyterLab];
    Plot-- Save plotted data --> DataHandler;
    style MeasurementDataclass fill:#bbf,stroke:#f66,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style Parameters fill:#bbf,stroke:#f66,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
```

### Measurement Dataclass
```mermaid
flowchart TD;
    subgraph Standard Data
        MeasurementDataclass-->filepath1;
        MeasurementDataclass-->CustomDataImporter[Handle custom data like confocal etc.];
        CustomDataImporter-->data1;
        MeasurementDataclass-->params1;
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

### Parameters

The `Parameters` dataclass in `parameters.py` contains the attributes about which computer is used and where the data is
stored. The code will automatically detect any VPN connection, and adjust its save location accordingly (Note that you 
cannot save to kernix when connected remotely).

| Attribute              | Explanation                                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------------|
| `lab_computer_name`    | Name of lab computer, use `os.environ["COMPUTERNAME"]` (eg. PCKK022)                                             |
| `remote_datafolder`    | Folder to connect to when running analysis remotely (eg. over VPN) (default: `\\kernix\qudiamond\Data`)          |
| `remote_output_folder` | Folder to place output images when running remotely (eg. over VPN) (default: `$USER\Documents\QudiHiraAnalysis`) |
| `local_datafolder`     | Folder to connect to when running  locally (default: `Z:\Data`)                                                  |
| `local_output_folder`  | Folder to place output images when running locally (default: `Z:\QudiHiraAnalysis`)                              |

## Examples

### Plot all confocal images

```python
import matplotlib.pyplot as plt
from src.data_handler import DataHandler

data_handler = DataHandler(measurement_folder="20220621_FR0612-F2-2S6_uhv")
confocal_list = data_handler.load_measurements_into_dataclass_list(measurement_str="Confocal")

fig, ax = plt.subplots(nrows=10)

for idx, confocal in enumerate(confocal_list):
    ax[idx].imshow(confocal.data)
    ax[idx].set_title(f"Laser power = {confocal.get_param_from_filename(unit='mW')}")

data_handler.save_figures(fig, filename="compare_confocals_at different_laser_powers")
```

See [ExampleNotebook.ipynb](ExampleNotebook.ipynb) for more examples.

### Fits available

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

## Getting Started

### Prerequisites

Latest version of the [conda](https://docs.conda.io/en/latest/miniconda.html) package manager.

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

where `xx` is either `win10` or `osx`.

#### Activate environment

```shell
conda activate qudi-hira-analysis
```

#### Add conda environment to Jupyter kernel

```shell
python -m ipykernel install --user --name=qudi-hira-analysis
```

### Start the analysis

```shell
jupyter lab
```

## Makefile options

The Makefile is configured to generate a variety of outputs:

+ `make pdf` : Converts all notebooks to PDF (requires LaTeX backend)
+ `make html`: Converts all notebooks to HTML files
+ `make py`  : Converts all notebooks to Python files (useful for VCS)
+ `make all` : Sequentially runs all the notebooks in folder

To use the `make` command on Windows you can install [Chocolatey](https://chocolatey.org/install), then
run `choco install make`
