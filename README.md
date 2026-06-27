# Darcy-PINN

This repository contains the source code accompanying our paper on **Self-Adaptive Physics-Informed Neural Networks for Forward and Inverse Problems in Heterogeneous Porous Media Flow** for forward and inverse Darcy flow problems. The repository includes implementations for forward and inverse PINNs, visualization scripts, and error analysis tools used in the numerical experiments.

# Repository Structure

* Forward_PINN/
  Implementation of the forward PINNs for the three permeability configurations considered in the paper. These scripts generate synthetic velocity and pressure observations used in the inverse PINN experiments.

* Inverse_PINN/
  Implementation of the inverse PINNs for reconstructing the permeability field from synthetic observations. Multiple regularization strategies are included for each problem.

* Plot_Speed_Pressure/
  Scripts for generating contour plots of the speed magnitude and pressure fields from FEM, forward PINN, or inverse PINN solution data.

* Permeability_Comparison/
  Scripts for visualizing and comparing the reconstructed permeability fields with the corresponding ground-truth permeability distributions.

* Error_Calculation/
  Scripts for computing quantitative error metrics (MAE, Relative (L^2), and RMSE) by comparing PINN predictions with the corresponding finite element (FEM) reference solutions.

# Numerical Experiments

The repository contains implementations for three permeability configurations:

1. Homogeneous permeability.
2. Single permeability anomaly.
3. Two permeability anomalies.

Each experiment is performed over 10 independent runs using random seeds 1–10 to assess the robustness and statistical consistency of the proposed methods.

# Requirements

The code is implemented in Python 3 using:

* PyTorch
* NumPy
* Pandas
* Matplotlib
* SciPy

Detailed instructions for running each component are provided in the corresponding subdirectory.
