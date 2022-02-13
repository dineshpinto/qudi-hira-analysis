# qudiamond-analysis

 A reproducible and transparent toolkit to analyze experimental data from an imaging magnetometer in diamond.
 
**Transparency** is achieved using Jupyter notebooks, which mix analysis code and figures along with written texts. The toolkit itself is built entirely on free and open-source software (FOSS).

**Reproducibility** is achieved using automated build tools (GNU Make) and environment metadata storage. Two lines of code are sufficient to reproduce all analyzed data and figures.

This license of this project is located in the top level folder under `LICENSE`. Some specific files contain their individual licenses in the file header docstring.

## Layout
+ JupyterLab notebooks are stored in `notebooks/`
+ Reusable code is stored in `src/`
  1. `io.py` Reading and writing data and figures
  2. `preprocessing.py` Processing raw counts into usable data
  3. `fitting.py` Set of (semi)-automated fitting routines
  4. `qudi_fit_wrapper.py` Set of fitting routines from the [`qudi`](https://github.com/Ulm-IQO/qudi) project
+ Utilities such as automated copy scripts, conda envs etc. are stored in `tools/`

## Setup 

### 1. Creating the conda environment
```
conda env create -f tools/conda-environment.yml
```

### 2. Add conda env to jupyter kernel
```
python -m ipykernel install --user --name=analysis
```

### 3. Enter environment
```
conda activate analysis
```

### Notes
- If exporting environments: ```conda env export --no-builds > tools/conda-env.yml```


## Makefile options
The Makefile is configured to generate a variety of outputs:

1. `make pdf` : Converts notebooks to PDF using LaTeX
2. `make html`: Converts notebooks to HTML files
3. `make py`  : Converts notebooks to Python files (useful for VCS)
4. `make all` : Sequentially runs all the notebooks in folder
