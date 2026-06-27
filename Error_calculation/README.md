# Error Calculation

This directory contains the script used to compute quantitative error metrics for the numerical experiments presented in our paper.

# File
Error_calculation.py – Computes the error between the predicted solution and the corresponding finite element (FEM) reference solution.

# Input

The script requires:

* The corresponding FEM reference solution (porus_1K, porus_2K, or porus_3K).
* A collection of predicted solution files containing:
Computational coordinates (x, y)
Velocity components (u_x, u_y)
Pressure (p)

The predicted solution files may be obtained from:

The 10 independent forward PINN runs.
The 10 independent runs of any inverse PINN regularization strategy.

The input filenames and the number of runs are specified directly within the script and can be modified as needed.

# Computed Error Metrics

For each solution variable (u_x, u_y, and p), the script computes:

Mean Absolute Error (MAE)
Relative (L^2) Error
Root Mean Squared Error (RMSE)

The error metrics are evaluated for each of the 10 runs and subsequently averaged to report the mean and standard deviation across all runs.

# Output

The script generates:

A CSV file containing the error metrics for every run.
A summary CSV file reporting the mean and standard deviation of each error metric over all runs.

The mean and standard deviation are also printed to the terminal.

# Requirements

The script requires:

Python 3.10 or later
NumPy
Pandas

# Notes

The script is designed to evaluate both forward and inverse PINN predictions against the corresponding FEM reference solution. By default, the statistics are computed over 10 independent runs, but the number of runs and the input filenames can be easily modified within the script.
