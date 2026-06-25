This folder contains the implementations of the forward Physics-Informed Neural Networks (PINNs) used in the numerical experiments presented in the paper.

Three forward cases are included:

ForwardPINN_1K.py – Homogeneous permeability field.
ForwardPINN_2K.py – Single permeability anomaly embedded in a homogeneous background.
ForwardPINN_3K.py – Two permeability anomalies embedded in a homogeneous background.

Each implementation solves the steady-state Darcy flow equations using a physics-informed neural network to predict the velocity and pressure fields.

Output

After training, each script exports the predicted solution in CSV format containing

x : x-coordinate
y : y-coordinate
u_x : x-component of the Darcy velocity
u_y : y-component of the Darcy velocity
p : pressure

The trained model parameters are also saved for subsequent inverse PINN experiments.

Requirements

The implementations require

Python 3.10+
PyTorch
NumPy
Pandas
Notes
All experiments use the same network architecture and training strategy described in the accompanying paper.
Random seeds are fixed to ensure reproducibility.
The generated forward solutions serve as synthetic observations for the inverse PINN experiments.
