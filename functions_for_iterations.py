def advect_phi(phi, u, v, dx, dy, dt):
    """
    Advection of the level set field using velocity components.
    """
    # CFL stability constraint
    cfl_dt = min(dx / (np.abs(u).max() + 1e-6), dy / (np.abs(v).max() + 1e-6))
    dt = min(dt, cfl_dt)
    
    # Compute gradients
    dphi_dx = np.gradient(phi, dx, axis=1)
    dphi_dy = np.gradient(phi, dy, axis=0)
    
    # Update phi using advection equation
    phi_new = phi - dt * (u * dphi_dx + v * dphi_dy)
    return phi_new

def reinitialize_phi(phi, dx, dy, tau=0.1, iterations=50):
    """
    Reinitialize the level set field to maintain the signed distance property.
    """
    epsilon = 1e-6  # Small parameter to avoid division by zero
    for _ in range(iterations):
        # Compute gradients
        dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
        dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dy)
        grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2 + epsilon**2)
        
        # Reinitialization step
        sign_phi = phi / np.sqrt(phi**2 + epsilon**2)
        phi += tau * sign_phi * (1 - grad_phi)
    return phi

def heaviside(phi, epsilon=1e-6):
    """
    Regularized Heaviside step function:
    Smooth transition near phi = 0 to improve numerical stability.
    """
    return 0.5 * (1 + phi / np.sqrt(phi**2 + epsilon**2))

def update_density(phi, rho_light, rho_heavy):
    """
    Update density based on the level set field (phi) using the Heaviside function.
    Parameters:
    - phi: Level set field (scalar or array)
    - rho_light: Density of the light fluid
    - rho_heavy: Density of the heavy fluid
    Returns:
    - rho: Updated density field
    """
    H = heaviside(phi)
    rho = rho_light + (rho_heavy - rho_light) * H
    return rho

def update_vorticity(omega, u, v, rho, dx, dy, nu, g, dt):
    """
    Update vorticity field using the vorticity equation.
    """
    # Compute gradients
    d_omega_dx = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dx)
    d_omega_dy = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dy)
    d_rho_dx = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dx)
    
    # Compute terms
    advection = u * d_omega_dx + v * d_omega_dy
    laplacian_omega = (
        (np.roll(omega, -1, axis=1) - 2 * omega + np.roll(omega, 1, axis=1)) / dx**2 +
        (np.roll(omega, -1, axis=0) - 2 * omega + np.roll(omega, 1, axis=0)) / dy**2
    )
    diffusion = nu * laplacian_omega
    baroclinic = g * d_rho_dx / np.maximum(rho, 1e-6)  # Avoid division by zero
    
    # Time-stepping
    omega_new = omega + dt * (-advection + diffusion + baroclinic)
    return omega_new


def solve_poisson_sor(omega, dx, dy, tol=1e-3, max_iter=5000, omega_relax=1.5):
    """
    Solve the Poisson equation using SOR for the stream function.
    """
    Ny, Nx = omega.shape
    psi = np.zeros_like(omega)  # Initialize stream function
    
    for iteration in range(max_iter):
        psi_old = psi.copy()
        
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                psi[i, j] = (1 - omega_relax) * psi[i, j] + omega_relax / (2 / dx**2 + 2 / dy**2) * (
                    (psi[i+1, j] + psi[i-1, j]) / dx**2 +
                    (psi[i, j+1] + psi[i, j-1]) / dy**2 - omega[i, j]
                )
        
        # Boundary conditions
        psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0
        
        # Convergence check
        if np.linalg.norm(psi - psi_old, ord=np.inf) < tol:
            break
    return psi


def update_velocities(psi, dx, dy):
    u = np.gradient(psi, axis=1) / dy  
    v = -np.gradient(psi, axis=0) / dx  
    return u, v
