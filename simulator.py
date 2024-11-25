import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Setting up the grid
Lx, Ly = 3.0, 1.0  # domain size
Nx, Ny = 300, 100  # grid resolution
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

y_interface = Ly / 2

rho_light, rho_heavy = 1.0, 2.0
delta = 0.01 

def density_field(y):
    """
    Function to create the density field
    """
    rho = rho_light + (rho_heavy - rho_light) * 0.5 * (1 + np.tanh((y - y_interface) / delta))
    return rho

def level_set_field(y):
    """
    Function to create the level set field
    """
    phi = y - y_interface
    return phi

rho = density_field(Y)
phi = level_set_field(Y)

# Create a custom colormap (blue for light fluid, red for heavy fluid)
colors = ['blue', 'red']
cmap = ListedColormap(colors)

# Visualization
plt.figure(figsize=(15, 5))
plt.contour(X, Y, phi, levels=[0], colors='black', linewidths=1)  # Interface (phi = 0)
plt.imshow(rho, extent=(0, Lx, 0, Ly), origin='lower', cmap=cmap, alpha=0.7)
plt.title("Initial Density and Level Set Field")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
