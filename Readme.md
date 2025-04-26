# Canopy Structure Exhibits Linear and Nonlinear Links to Biome-Level Maximum Light Use Efficiency

This repository contains the code and data for the manuscript titled "Canopy Structure Exhibits Linear and Nonlinear Links to Biome-Level Maximum Light Use Efficiency".  
[Link to be updated upon publication]

# Repository Structure

## Data
- `EC_data.csv` - Eddy covariance data compiled from AmeriFlux, FLUXNET, and ICOS networks
- `site_coordinates/` - Directory containing geographic coordinates for flux tower sites
  - `ameriflux_coordinates.csv`
  - `fluxnet_coordinates.csv`
  - `icos_coordinates.csv`

## Code
- `utils.py` - Helper functions for:
  - Data preprocessing
  - Statistical analysis
  - Model fitting (including Holling Type II models)
- `plotting.py` - Visualization functions to reproduce all figures in the manuscript:
  - `plot_holling_relationship()` - Holling Type II model fitting plots
  - `plot_regression_comparison()` - Linear regression comparison plots
  - `plot_biome_boxplot()` - Biome-level distributions of variables
- `main.ipynb` - Jupyter notebook that runs all analyses and reproduces the results, tables, and figures

## Outputs
- Directory where all generated figures and tables are saved
- Figures are saved in PNG format at 300 DPI

# Reproducing the Results

## Environment Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate scientific_analysis
   ```

3. If Jupyter is not installed, install it:
   ```bash
   conda install jupyter
   ```

## Running the Analysis

1. Start a Jupyter notebook server:
   ```bash
   jupyter notebook
   ```

2. Open `main.ipynb` in the Jupyter interface.

3. Run all cells in the notebook to reproduce the analysis, tables, and figures.

# Key Dependencies

This code has been tested with the following package versions:

- Python 3.10
- cartopy 0.22.0
- geopandas 0.14.2
- matplotlib 3.8.2
- numpy 1.26.4
- pandas 2.1.3
- scipy 1.12.0
- statsmodels 0.14.1
- scikit-learn 1.4.2
- seaborn (latest)

# Data Sources

The eddy covariance data used in this study comes from:

- AmeriFlux
- FLUXNET
- ICOS

Please cite these data sources appropriately when using this code or reproducing these results.

# Citation

TBD

# Contact

dashtiahanga@wisc.edu