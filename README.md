# qudiamond-analysis

A python toolkit to analyze experimental data from an NV diamond atomic force microscope. The toolkit aims to make the data analysis transparent and reproducible.

**Transparency** is achieved using Jupyter notebooks, which mix analysis code and figures along with written texts. The toolkit itself is built entirely on free and open-source software (FOSS).

**Reproducibility** is achieved using automated build tools (GNU Make) and environment metadata storage. Two lines of code are sufficient to reproduce all analyzed data and figures.

This license of this project is located in `LICENSE.md`

## Logic
+ Jupyter notebooks are stored in `notebooks/`
+ Reusable code is stored in `src/`
  1. `io.py` Reading and writing data and figures
  2. `preprocessing.py` Processing raw counts into usable data
  3. `fitting.py` Set of (semi)-automated fitting routines
+ Utilities such as automated copy scripts, conda envs etc. are stored in `tools/`

## Setup Method 1

### 1. Creating the conda environment
```conda env create -f conda-environment.yml```

### 2. Interactive plotting in jupyter lab
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension
```

### Notes
- Use ```import ipympl``` in notebook
- If exporting environments: ```conda env export --no-builds -f conda-environment.yml```

## Setup Method 2

### 1. Conda env
```conda  create --name jupyterlab```
### 2. Activate env
```activate jupyterlab```
### 3. Conda base environment
```
conda install -c conda-forge python scipy numpy jupyterlab lmfit matplotlib pandas peakutils jupytext tqdm
pip install pySPM
```

### 4. Interactive plotting with jupyterlab
```
conda install -c conda-forge nodejs
pip install ipympl
pip install --upgrade jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension
```

### Notes
- Use `import ipympl` in notebook


## Makefile options
The Makefile is configured to generate a variety of outputs:

1. `make pdf` : Converts notebooks to PDF using LaTeX
2. `make html`: Converts notebooks to HTML files
3. `make py`  : Converts notebooks to Python files (useful for VCS)
4. `make all` : Sequentially runs all the notebooks in folder
