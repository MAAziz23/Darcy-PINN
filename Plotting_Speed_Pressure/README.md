# Speed and Pressure Plots

This directory contains the visualization script used to generate speed and pressure field plots for the numerical experiments presented in our paper.

# File
Plot_Speed_Pressure.py – Generates contour plots of the speed magnitude and pressure field from solution data.

# Input

The script accepts any solution data file containing the following quantities:

* Computational coordinates (x, y)
* Velocity components (u_x, u_y)
* Pressure (p)

The input file may be obtained from:

* A forward PINN output.
* A finite element (FEM) reference solution.
* An inverse PINN output.

Simply specify the desired input filename within the script to visualize the corresponding speed and pressure fields.

# Output

The script generates:

* A contour plot of the speed magnitude,
 [
 |\mathbf{u}|=\sqrt{u_x^2+u_y^2},
 ]
* A contour plot of the pressure field.

Both figures are saved as high-resolution PNG files and displayed on screen.

# Requirements

The script requires:

Python 3.10 or later
NumPy
Pandas
Matplotlib
SciPy

# Notes

The plotting settings, including interpolation, color mapping, fonts, and axis formatting, are configured to produce publication-quality figures consistent with those presented in the accompanying paper. The input filename is specified directly within the script and can be modified to visualize different datasets.
