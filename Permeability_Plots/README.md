# Permeability Comparison Plots

This directory contains the visualization script used to compare the reconstructed permeability fields obtained from the inverse PINN with the corresponding ground-truth permeability distributions for the three numerical test cases considered in our paper.

# File
K_plotting.py – Generates side-by-side comparisons of the true and reconstructed permeability fields for the homogeneous, single-anomaly, and two-anomaly permeability cases.

# Input

The script requires the reconstructed permeability files produced by the inverse PINN implementations. For visualization, a representative run is selected from the 10 independent runs performed for each regularization strategy. By default, the script uses:

* inverse_output_1K_noreg_run1.csv
* inverse_output_2K_L2TV_run1.csv
* inverse_output_3K_L2_run1.csv

Each input file contains:

* Computational coordinates (x, y)
* Predicted Darcy velocity (u_x, u_y)
* Predicted pressure (p)
* Reconstructed permeability (K)

# Output

The script generates side-by-side visual comparisons of:

* True permeability field
* Reconstructed permeability field

for the following permeability configurations:

* Homogeneous permeability
* Single permeability anomaly
* Two permeability anomalies

All figures are produced using a common color scale, equal aspect ratio, and consistent plotting settings to facilitate direct visual comparison across the three cases.

# Requirements

The script requires:

Python 3.10 or later
NumPy
Pandas
Matplotlib
SciPy

# Notes

The script automatically constructs the corresponding ground-truth permeability field for each case and compares it with the reconstructed permeability obtained from the inverse PINN. The plotted results correspond to representative runs selected from the 10 independent runs performed for each regularization strategy. The quantitative performance metrics reported in the accompanying paper are computed using all 10 runs. The input filenames are specified directly within the script and can be modified to visualize different runs or regularization strategies.
