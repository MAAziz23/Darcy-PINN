#!/usr/bin/env python
# coding: utf-8

# Uniform K comparison

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

GLOBAL_VMIN = 0.04  
GLOBAL_VMAX = 0.12  
N_TICKS = 5

plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22,
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

predicted_data = pd.read_csv('inverse_output_1K_noreg_run1.csv')
x = predicted_data['x'].values
y = predicted_data['y'].values
K_pred = predicted_data['K'].values

K_true = 0.05 * np.ones_like(x)

grid_x, grid_y = np.meshgrid(
    np.linspace(0, 1, 200),
    np.linspace(0, 1, 200)
)

K_true_grid = griddata((x, y), K_true, (grid_x, grid_y), method='linear')
K_pred_grid = griddata((x, y), K_pred, (grid_x, grid_y), method='cubic')

vmin = GLOBAL_VMIN
vmax = GLOBAL_VMAX

fig, axs = plt.subplots(1, 2, figsize=(18, 6))
tick_locations = np.arange(0, 1.1, 0.2)  

for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    
    ax.set_xticks(tick_locations)
    ax.set_yticks(tick_locations)
    
    ax.xaxis.set_major_formatter(FuncFormatter(smart_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(smart_formatter))

ticks = np.linspace(vmin, vmax, N_TICKS)

div1 = make_axes_locatable(axs[0])
cax1 = div1.append_axes("right", size="5%", pad=0.2)
im1 = axs[0].imshow(
    K_true_grid,
    extent=[0, 1, 0, 1],  
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axs[0].set_title(r"$\text{True } K(x,y)$")
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels([f"{tick:.3f}" for tick in ticks])

div2 = make_axes_locatable(axs[1])
cax2 = div2.append_axes("right", size="5%", pad=0.2)
im2 = axs[1].imshow(
    K_pred_grid,
    extent=[0, 1, 0, 1],  
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axs[1].set_title(r"$\text{Predicted } K(x,y)$")
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.set_ticks(ticks)
cbar2.set_ticklabels([f"{tick:.3f}" for tick in ticks])

plt.tight_layout()
plt.show()


# One anomaly K comparison

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter


GLOBAL_VMIN = 0.04   
GLOBAL_VMAX = 0.12   
N_TICKS = 5


plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22,
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


predicted_data = pd.read_csv('inverse_output_2k_L2TV_run1.csv')
x = predicted_data['x'].values
y = predicted_data['y'].values
K_pred = predicted_data['K'].values  

K_true = np.where((0.4 <= x) & (x <= 0.6) & (0.4 <= y) & (y <= 0.6), 0.1, 0.05)


grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
K_true_grid = griddata((x, y), K_true, (grid_x, grid_y), method='cubic')
K_pred_grid = griddata((x, y), K_pred, (grid_x, grid_y), method='cubic')

vmin = GLOBAL_VMIN
vmax = GLOBAL_VMAX


fig, axs = plt.subplots(1, 2, figsize=(18, 6))

tick_locations = np.arange(0, 1.1, 0.2)  

for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    
    ax.set_xticks(tick_locations)
    ax.set_yticks(tick_locations)
    
    ax.xaxis.set_major_formatter(FuncFormatter(smart_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(smart_formatter))

ticks = np.linspace(vmin, vmax, N_TICKS)

div1 = make_axes_locatable(axs[0])
cax1 = div1.append_axes("right", size="5%", pad=0.2)
im1 = axs[0].imshow(
    K_true_grid,
    extent=[0, 1, 0, 1], 
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axs[0].set_title(r"$\text{True } K(x, y)$")
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels([f"{tick:.3f}" for tick in ticks])

div2 = make_axes_locatable(axs[1])
cax2 = div2.append_axes("right", size="5%", pad=0.2)
im2 = axs[1].imshow(
    K_pred_grid,
    extent=[0, 1, 0, 1], 
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axs[1].set_title(r"$\text{Predicted } K(x, y)$")
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.set_ticks(ticks)
cbar2.set_ticklabels([f"{tick:.3f}" for tick in ticks])

plt.tight_layout()
plt.show()


# Two anomalies K comparison

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

GLOBAL_VMIN = 0.04  
GLOBAL_VMAX = 0.12  
N_TICKS = 5

plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 22,
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

predicted_data = pd.read_csv('inverse_output_3K_L2_run1.csv')
x = predicted_data['x'].values
y = predicted_data['y'].values
K_pred = predicted_data['K'].values 


K_true = np.ones_like(x) * 0.05  

anomaly1_mask = (0.1 <= x) & (x <= 0.3) & (0.1 <= y) & (y <= 0.3)

anomaly2_mask = (0.7 <= x) & (x <= 0.9) & (0.7 <= y) & (y <= 0.9)

K_true[anomaly1_mask] = 0.1
K_true[anomaly2_mask] = 0.1


grid_x, grid_y = np.meshgrid(
    np.linspace(0, 1, 200),
    np.linspace(0, 1, 200)
)

K_true_grid = griddata((x, y), K_true, (grid_x, grid_y), method='nearest')
K_pred_grid = griddata((x, y), K_pred, (grid_x, grid_y), method='nearest')


vmin = GLOBAL_VMIN
vmax = GLOBAL_VMAX


fig, axs = plt.subplots(1, 2, figsize=(18, 6))

tick_locations = np.arange(0, 1.1, 0.2)  

for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    
    ax.set_xticks(tick_locations)
    ax.set_yticks(tick_locations)
    
    ax.xaxis.set_major_formatter(FuncFormatter(smart_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(smart_formatter))

ticks = np.linspace(vmin, vmax, N_TICKS)

div1 = make_axes_locatable(axs[0])
cax1 = div1.append_axes("right", size="5%", pad=0.2)
im1 = axs[0].imshow(
    K_true_grid,
    extent=[0, 1, 0, 1],
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axs[0].set_title(r"$\text{True } K(x,y)$")
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels([f"{tick:.3f}" for tick in ticks])

div2 = make_axes_locatable(axs[1])
cax2 = div2.append_axes("right", size="5%", pad=0.2)
im2 = axs[1].imshow(
    K_pred_grid,
    extent=[0, 1, 0, 1],
    origin='lower',
    aspect='equal',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)
axs[1].set_title(r"$\text{Predicted } K(x,y)$")
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.set_ticks(ticks)
cbar2.set_ticklabels([f"{tick:.3f}" for tick in ticks])

plt.tight_layout()
plt.show()


# In[ ]:




