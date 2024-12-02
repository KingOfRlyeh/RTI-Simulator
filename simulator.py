import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Domain setup
Lx, Ly = 1.0, 3.0  # Domain dimensions
Nx, Ny = 10, 30  # Grid resolution
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)


# Physical parameters
rho_light, rho_heavy = 1.0, 3.0  # Fluid densities
delta = 0.01  # Interface thickness
nu = 0.01  # Viscosity
g = 9.8  # Gravitational acceleration
dt = 0.00001  # Time step
T = 0.001 # Total simulation time
frames = int(T / dt) # number of frames

# Initial conditions
y_interface = Ly / 2  # Interface position
phi = Y - y_interface + 0.05 * np.sin(2 * np.pi * X / Lx)  # Perturbed interface
rho = rho_light + (rho_heavy - rho_light) * 0.5 * (1 + np.tanh((Y - y_interface) / delta))
u = np.zeros_like(X)  # Horizontal velocity
v = np.zeros_like(Y)  # Vertical velocity
omega = np.zeros_like(X)  # Vorticity

# Helper functions
def advect_phi(phi, u, v, dx, dy, dt):
    dphi_dx = np.gradient(phi, axis=1) / dx
    dphi_dy = np.gradient(phi, axis=0) / dy
    phi_new = phi - dt * (u * dphi_dx + v * dphi_dy)
    return phi_new

def reinitialize_phi(phi, dx, dy, tau=0.1, iterations=50):
    epsilon = 1e-6
    for _ in range(iterations):
        dphi_dx = np.gradient(phi, axis=1) / dx
        dphi_dy = np.gradient(phi, axis=0) / dy
        grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2 + epsilon**2)
        sign_phi = phi / np.sqrt(phi**2 + epsilon**2)
        phi -= tau * sign_phi * (grad_phi - 1)
    return phi

def heaviside(phi, epsilon=1e-6):
    return 0.5 * (1 + phi / np.sqrt(phi**2 + epsilon**2))

def update_density(phi, rho_light, rho_heavy):
    H = heaviside(phi)
    return rho_light + (rho_heavy - rho_light) * H

def update_vorticity(omega, u, v, rho, dx, dy, nu, g, dt):
    d_omega_dx = np.gradient(omega, axis=1) / dx
    d_omega_dy = np.gradient(omega, axis=0) / dy
    d_rho_dx = np.gradient(rho, axis=1) / dx

    advection = u * d_omega_dx + v * d_omega_dy
    laplacian_omega = (
        (np.roll(omega, -1, axis=1) - 2 * omega + np.roll(omega, 1, axis=1)) / dx**2 +
        (np.roll(omega, -1, axis=0) - 2 * omega + np.roll(omega, 1, axis=0)) / dy**2
    )
    diffusion = nu * laplacian_omega
    baroclinic = g * d_rho_dx / np.maximum(rho, 1e-6)

    omega_new = omega + dt * (-advection + diffusion + baroclinic)
    omega_new[0, :] = omega_new[-1, :] = omega_new[:, 0] = omega_new[:, -1] = 0  # Reset at boundaries
    return omega_new

def solve_poisson_sor(omega, dx, dy, tol=1e-2, max_iter=5000, omega_relax=1.5):
    psi = np.zeros_like(omega)
    for _ in range(max_iter):
        psi_old = psi.copy()
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                psi[i, j] = (1 - omega_relax) * psi[i, j] + omega_relax / (2 / dx**2 + 2 / dy**2) * (
                    (psi[i+1, j] + psi[i-1, j]) / dx**2 +
                    (psi[i, j+1] + psi[i, j-1]) / dy**2 - omega[i, j]
                )
        psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0  # Boundary conditions
        if np.linalg.norm(psi - psi_old, ord=np.inf) < tol:
            break
    return psi

def update_velocities(psi, dx, dy):
    u = np.gradient(psi, axis=1) / dy
    v = -np.gradient(psi, axis=0) / dx
    return u, v

# Visualization setup
fig, ax = plt.subplots(figsize=(6, 8))

def init():
    ax.imshow(rho, extent=(0, Lx, 0, Ly), origin='lower', cmap='coolwarm', alpha=0.7)
    return []

def update(frame):
    global phi, rho, u, v, omega

    phi = advect_phi(phi, u, v, dx, dy, dt)
    phi = reinitialize_phi(phi, dx, dy, tau=0.1, iterations=10)
    rho = update_density(phi, rho_light, rho_heavy)
    omega = update_vorticity(omega, u, v, rho, dx, dy, nu, g, dt)
    psi = solve_poisson_sor(omega, dx, dy, tol=1e-2)
    u, v = update_velocities(psi, dx, dy)

    ax.clear()
    ax.imshow(rho, extent=(0, Lx, 0, Ly), origin='lower', cmap='coolwarm', alpha=0.7)
    ax.contour(X, Y, phi, levels=[0], colors='black', linewidths=2)
    ax.quiver(X[::4, ::4], Y[::4, ::4], u[::4, ::4], v[::4, ::4], scale=10, color='white')
    ax.set_title(f"Time: {frame * dt:.4f} seconds")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return []

# Animation
anim = animation.FuncAnimation(fig, update, frames=frames, interval=1, init_func=init, blit=False)

# Display animation
HTML(anim.to_jshtml())
