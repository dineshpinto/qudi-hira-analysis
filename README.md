# data_analysis

A python toolkit to analyze color center experimental data. The toolkit aims to make the data analysis **transparent** and **reproducible**.

Transparency is achieved using Jupyter notebooks, which mix analysis code and figures along with written texts. The toolkit itself is built entirely on free and open-source software (FOSS).

Reproducibility is achieved using automated build tools (GNU Make) and environment metadata storage. Two lines of code are sufficient to reproduce _all_ analyzed data and figures.

## Data processing logic
1. `raw/` contains all the raw experimental data. The folder structure is the same as `kernix/`. The top-level folder is labeled as `YYMMDD_SAMPLE_SURFACE_CONDITION/`, eg. `180110_9002_NC60_LT` for a **9002** diamond coated with **N@C60** and measured at **low-temperature** conditions.
2. `processing_YYMMDD_SAMPLE_SURFACE_CONDITION.ipynb` processes raw data into a python dictionary stored as a pickle file in `data/YYMMDD_SAMPLE_SURFACE_CONDITION/`.
3. `plotting_YYMMDD_SAMPLE_SURFACE_CONDITION.ipynb` inputs the processed data and outputs figures stored in `figures/_YYMMDD_SAMPLE_SURFACE_CONDITION/`.
4. Reusable code is stored in `src/`
  1. `io.py` Reading and writing data and figures
  2. `preprocessing.py` Processing raw counts into usable data
  3. `fitting.py` Set of (semi)-automated fitting routines

## How to use it
1. Place the raw data from `kernix/diamond_afm/data/` into `raw/`, taking account of the different top level folder naming convention.
2. Generate processed data and figures either automatically or manually:
  1. **Automated:** Run the Makefile within `develop/` using `make all`.
  2. **Manual:** Use the `Run all cells` option within the notebooks (`processing_<...>.ipynb` has to be run before `plotting_<...>.ipynb`).
3. The processed data is stored in `data/` and the output figures in `figures/`

## Makefile options
The Makefile is configured to generate a variety of outputs:

1. `make pdf` : Converts notebooks to PDF using LaTeX
2. `make html`: Converts notebooks to HTML files
3. `make py`  : Converts notebooks to Python files (useful for VCS)
4. `make all` : Sequentially runs all the notebooks in folder

## Packages
`package-list.txt` contains a list of all conda packages used. A conda environment can be created with these packages using `conda env create -f data_analysis_env_linux.yml`.
