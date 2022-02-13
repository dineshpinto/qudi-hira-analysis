# qudiamond-analysis

 A reproducible and transparent toolkit to analyze experimental data from an imaging magnetometer in diamond.
 
**Transparency** is achieved using Jupyter notebooks, which mix analysis code and figures along with written texts. The toolkit itself is built entirely on free and open-source software (FOSS).

**Reproducibility** is achieved using automated build tools (GNU Make) and environment metadata storage. Two lines of code are sufficient to reproduce all analyzed data and figures.

This license of this project is located in the top level folder under `LICENSE`. Some specific files contain their individual licenses in the file header docstring.

## Layout
+ JupyterLab notebooks are stored in `notebooks/`
+ Reusable code is stored in `src/`
  + `io.py` Reading and writing data and figures
  + `preprocessing.py` Processing raw counts into usable data
  + `fitting.py` Set of (semi)-automated fitting routines
  + `qudi_fit_wrapper.py` Wrapper around fitting methods from the [`qudi`](https://github.com/Ulm-IQO/qudi) project
+ Utilities such as automated copy scripts, conda envs etc. are stored in `tools/`

## Prerequisites
- Python 3.10 or higher
- Conda 4.11 or higher

## Getting Started 

### Clone the repository

#### With Git
```shell
git clone https://github.com/dineshpinto/qudiamond-analysis.git
```

#### With Github CLI
```shell
gh repo clone dineshpinto/qudiamond-analysis
```

### Installing dependencies

#### Creating the conda environment
```shell
conda env create -f tools/conda-environment.yml
```

#### Activate environment
```shell
conda activate analysis
```

#### Add conda environment to jupyter kernel
```shell
python -m ipykernel install --user --name=analysis
```

### Start the analysis
```shell
jupyter lab
```

### Notes
- If exporting environments: ```conda env export --no-builds > tools/conda-env.yml```


## Makefile options
The Makefile is configured to generate a variety of outputs:

+ `make pdf` : Converts notebooks to PDF using LaTeX
+ `make html`: Converts notebooks to HTML files
+ `make py`  : Converts notebooks to Python files (useful for VCS)
+ `make all` : Sequentially runs all the notebooks in folder
