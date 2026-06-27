#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import griddata

plt.rcParams.update({
    'axes.titlepad':12,
    'font.size': 30,
    'axes.titlesize': 34,
    'axes.labelsize': 16,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 32,
    'figure.titlesize': 34,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{bm}',
})


def smart_formatter(x, _):
    x = round(x, 6)  
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    else:
        return f"{x:.6g}"  

data = pd.read_csv('forward_output_3K_run1.csv')

x  = data.iloc[:, 0].values
y  = data.iloc[:, 1].values
u_x  = data.iloc[:, 2].values
u_y  = data.iloc[:, 3].values
p  = data.iloc[:, 4].values

grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
speed    = griddata((x, y), np.sqrt(u_x**2 + u_y**2), (grid_x, grid_y), method='cubic')
pressure = griddata((x, y), p, (grid_x, grid_y), method='cubic')

# Plot speed field

plt.figure(figsize=(10, 8))
contourf_speed = plt.contourf(grid_x, grid_y, speed, levels=10, cmap='jet')

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))

cbar = plt.colorbar(contourf_speed, fraction=0.08, pad=0.04)
cbar.ax.tick_params(width=2)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))
plt.title(r'Forward: $\lVert\boldsymbol{u_\theta}\rVert$')
plt.grid(False)
plt.savefig("Velocity_forward_3K.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot pressure field

plt.figure(figsize=(10, 8))
contourf_pressure = plt.contourf(grid_x, grid_y, pressure, levels=10, cmap='jet')

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))

cbar = plt.colorbar(contourf_pressure, fraction=0.08, pad=0.04)
cbar.ax.tick_params(width=2)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(smart_formatter))

plt.title(r'Forward: $p_\theta$')
plt.grid(False)
plt.savefig("Pressure_forward_3K.png", dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




