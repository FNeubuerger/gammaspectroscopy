# gammaspectroscopy
## Overview

This repository contains tools and scripts for gamma spectroscopy analysis. Gamma spectroscopy is a technique used to measure and analyze the energy spectra of gamma-ray sources.

## Features

- **Data Extraction**: Extract data from various file formats including text and PDF.
- **Peak Detection**: Identify peaks in gamma spectra using advanced signal processing techniques.
- **Isotope Identification**: Match detected peaks to known isotopes and calculate confidence scores.
- **Visualization**: Generate plots and visualizations of gamma spectra for analysis.
- **Background Correction**: Apply background correction to improve peak detection accuracy.
- **Automated Analysis**: Batch process multiple spectra files for automated analysis.
- **Customizable**: Easily extend and customize the analysis pipeline with additional functions.

## Contents

This repository includes the following directories and files:

- **notebooks/**: Jupyter notebooks for data analysis and visualization.
    - `analyse_spectra.ipynb`: Notebook for analyzing gamma spectra and identifying isotopes.
    - `visualize_results.ipynb`: Notebook for visualizing the results of the analysis.
    - `analyse_spectra_outliers.ipynb`: Notebook for analyzing outliers in the spectra data.
    - `test_read_table.ipynb`: Notebook for testing the reading of tables from PDF files.
- **src/**: Source code for various utility functions used in the analysis.
    - `utils.py`: Utility functions for data processing and analysis.
- **data/**: Directory for storing raw and processed data files.
- **results/**: Directory for storing analysis results and output files.
- **README.md**: This file, providing an overview and documentation of the repository.
- **LICENSE**: The license file for the repository.
- **.gitignore**: Git ignore file specifying files and directories to be ignored by Git.

## Getting Started

To get started with the analysis, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Run the Jupyter notebooks in the `notebooks/` directory to perform the analysis and visualize the results.

For detailed instructions, refer to the individual notebooks and the comments within the code.