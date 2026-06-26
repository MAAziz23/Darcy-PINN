# Forward PINN

This folder contains the implementations of the forward Physics-Informed Neural Networks (PINNs) used in the numerical experiments presented in our paper. The forward PINNs solve the steady-state Darcy flow equations and predict the Darcy velocity and pressure fields for three different permeability configurations.

## Files

The folder contains three Python scripts corresponding to the three forward test cases:

ForwardPINN_1K.py – Homogeneous permeability field.
ForwardPINN_2K.py – Single permeability anomaly embedded in a homogeneous background.
ForwardPINN_3K.py – Two permeability anomalies embedded in a homogeneous background.

All three implementations use the same network architecture, training strategy, and boundary conditions. The only difference between the scripts is the prescribed permeability distribution.

## Input

Each script requires its corresponding porous media data file (`porus_1K`, `porus_2K`, or `porus_3K`), which contains the computational coordinates together with the corresponding finite element (FEM) reference solution.

The input file contains the following information:

Column 1: (x)-coordinate
Column 2: (y)-coordinate
Column 3: FEM velocity component (u_x)
Column 4: FEM velocity component (u_y)
Column 5: FEM pressure (p)

Only the coordinate data ((x,y)) is used during PINN training to define the collocation points and boundary conditions. The FEM solution ((u_x,u_y,p)) is not used for training. It is used exclusively for quantitative error analysis and comparison with the PINN predictions after training.

## Output

Each script generates:

A trained PINN model (`.pth` file).
A CSV file containing:

  *`x` : x-coordinate
  * `y` : y-coordinate
  * `u_x` : predicted Darcy velocity in the (x)-direction
  * `u_y` : predicted Darcy velocity in the (y)-direction
  * `p` : predicted pressure

The generated forward solutions are used as synthetic observations for the inverse PINN experiments.

## Requirements

The implementations require:

* Python 3.10 or later
* PyTorch
* NumPy
* Pandas

## Reproducibility

Random seeds are fixed in all experiments to ensure reproducibility. The implementations follow the network architecture, training procedure, and numerical setup described in the accompanying paper.
