def advect_phi(phi, u, v, dx, dy, dt):
    """
    Advection of level set field.
    
    Parameters:
    - phi: Initial level set field
    - u, v: Velocity components
    - dx, dy: Grid spacing
    - dt: time spacing
    
    Returns:
    - phi_new: Advected level set field
    """
    # Compute gradients
    dphi_dx = np.gradient(phi, dx, axis=1)  
    dphi_dy = np.gradient(phi, dy, axis=0)  

    # Advection equation
    phi_new = phi - dt * (u * dphi_dx + v * dphi_dy)
    return phi_new

def reinitialize_phi(phi, dx, dy, tau, iterations):
    """
    Reinitialize the level set field to maintain signed distance property.
    
    Parameters:
    - phi: Initial level set field
    - dx, dy: Grid spacing
    - tau: Artificial time step for reinitialization
    - iterations: Number of iterations to perform
    
    Returns:
    - phi: Reinitialized level set field
    """
    epsilon = 1e-6  # Small parameter to avoid division by zero

    for _ in range(iterations):
        # Compute gradients
        dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
        dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dy)
        
        # Gradient magnitude
        grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2)
        
        # Sign function
        sign_phi = phi / np.sqrt(phi**2 + epsilon**2)
        
        # Update phi
        phi += tau * sign_phi * (1 - grad_phi)
    
    return phi

def heaviside(phi):
    """
    Heaviside step function:
    - H(phi) = -1 for phi < 0
    - H(phi) = 1 for phi > 0
    - H(phi) = 0 for phi = 0
    """
    return np.where(phi < 0, 0, np.where(phi > 0, 1, 0))

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
    
    Parameters:
    - omega: Current vorticity field
    - u, v: Velocity components
    - rho: Density field
    - dx, dy: Grid spacing
    - nu: Kinematic viscosity
    - g: Gravitational acceleration
    - dt: Time step size
    
    Returns:
    - omega_new: Updated vorticity field
    """
    # Compute gradients
    d_omega_dx = (np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)) / (2 * dx)
    d_omega_dy = (np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)) / (2 * dy)
    d_rho_dx = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dx)
    
    # Advection term
    advection = u * d_omega_dx + v * d_omega_dy
    
    # Diffusion term
    laplacian_omega = (
        (np.roll(omega, -1, axis=1) - 2 * omega + np.roll(omega, 1, axis=1)) / dx**2 +
        (np.roll(omega, -1, axis=0) - 2 * omega + np.roll(omega, 1, axis=0)) / dy**2
    )
    diffusion = nu * laplacian_omega
    
    # Baroclinic term
    baroclinic = (g / rho) * d_rho_dx
    
    # Time-stepping
    omega_new = omega + dt * (-advection + diffusion + baroclinic)
    
    return omega_new

def solve_poisson_sor(omega, dx, dy, tol=1e-2, max_iter=10000, omega_relax=1.5):
    """
    Solves the Poisson equation using Successive Over-Relaxation (SOR).

    Parameters:
    - omega: Vorticity field (2D array)
    - dx, dy: Grid spacings
    - tol: Convergence tolerance
    - max_iter: Maximum number of iterations
    - omega_relax: Relaxation factor
    
    Returns:
    - psi: Stream function (2D array)
    """
    Ny, Nx = omega.shape
    psi = np.zeros_like(omega)  # Initialize stream function
    
    for iteration in range(max_iter):
        psi_old = psi.copy()
        
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                psi[i, j] = (1 - omega_relax) * psi[i, j] + omega_relax / (2 / dx**2 + 2 / dy**2) * (
                    (psi[i+1, j] + psi[i-1, j]) / dx**2 +
                    (psi[i, j+1] + psi[i, j-1]) / dy**2 -
                    omega[i, j]
                )
        
        # Check for convergence
        if np.linalg.norm(psi - psi_old, ord=np.inf) < tol:
            print(f"Converged in {iteration} iterations")
            break
    else:
        print("Failed to converge")
    
    return psi

def update_velocities(psi, dx, dy):
    u = np.gradient(psi, axis=1) / dy  
    v = -np.gradient(psi, axis=0) / dx  
    return u, v
