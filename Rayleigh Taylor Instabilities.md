---
created: 2024-11-18T20:48
updated: 2024-11-24T10:55
---

# Set up
1. Initialize domain
		define a grid to be the domain
	- Initialize the density field $\rho(x,y)$, this is a discrete scalar field
	- Initialize velocity field $\vec{v}(x,y)=\langle u,v \rangle$
	-  Initialize level set field $\phi(x,y)$ (explained below)
2. Physical parameters
	- $\rho_{\text{light}},\rho_{\text{heavy}}$: Densities of the two fluids
		- notice that $\rho(x,y)$ is the scalar field such that every $\rho$ above the boundary is $\rho_{\text{heavy}}$ and every $\rho$ above the boundary is $\rho_{\text{light}}$
	- $g$: gravity
	- $\nu$: kinematic viscosity
# Computational Method
We will use the **Level Set Method** for Rayleigh-Taylor Instabilities

This involves only computing the *boundary* and how it evolves, which should make visualization pretty nice!
## Set up Scalar field
Basically, we're creating a scalar field $\phi(x,y,t)$ and defining the interface as places where $\phi =0$ for each $t$

So, we'll initialize the interface as $\phi(x,y)=0$ with:
- $\phi>0$ being the denser fluid on the bottom 
- $\phi<0$ being the lighter fluid on the top

Notice that as $\phi$ changes over time, the places where $\phi=0$ will mark the boundary between the two fluids

So we could for example, create a $n\times m$ grid representing the simulation space and set $\phi(x,y)=y-y_{\text{interface}}$ this makes it so that everything "below" the interface is negative

# Governing Equations
The velocity field will drive the change in $\phi$
Oh boy, now we gotta do navier stokes.

- **Navier stokes for $\vec{v}$:**
$$\frac{{\partial \vec{v}}}{\partial t}+(\vec{v}\cdot \nabla)\vec{v}=-\frac{1}{\rho}\nabla \rho + \nu \nabla^{2}\vec{v}+\vec{g}$$
- **Continuity** for incompressible flow:
$$\nabla \cdot \vec{v}=0$$
(btw all that  equation is saying is that the fluid can't "diverge"... i.e. it cant just decide to make a vacuum bubble. It also can't be compressed.)

- **Level Set Equation**, to track the interface
$$\frac{{\partial \phi}}{\partial t}+\vec{v}\cdot \nabla \phi=0$$
- **Density evolution**, basically applies the changes in the level set field to the density field. Notice that this setup *does not* allow the "dissolution" of one liquid into another. it'll be more like water and oil
		$$\rho(x,y,t)=\rho_{\text{light}}+(\rho_{\text{heavy}}-\rho_{\text{light}})H(\phi(x,y,t))$$
		where $H$ is the heavyside function. Basically if input is < 0 then it returns -1, if input is >0 then it returns 1, and if it's 0 it's 0
		
- **Vorticity Calculation**
	- define vorticity as $\omega=\frac{{\partial v}}{\partial x}-\frac{{\partial u}}{\partial y}$
	-  With this info we can solve this for $\omega$ $$\frac{{\partial \omega}}{\partial t}+\vec{v}\cdot \nabla \omega =\nu \nabla^{2}\omega+\frac{g}{\rho} \frac{{\partial \rho}}{\partial 2}$$
**Velocity Recovery**:
- Solve the **Poisson equation** for the stream function $$\nabla^{2}\psi=-\omega$$
- Derive velocities $$u=\frac{{\partial \psi}}{\partial y}, v=-\frac{{\partial \psi}}{\partial x}$$

# Numerical Implementation

1. **Initialize Fields**
	1. **Grid**: Create a uniform 2d grid $(x,y)$.
	2. **Density Field**: Use a smooth transition for initial $\rho(x,y)$ you could also use heaviside if you wanted to.
$$\rho(x,y,0)=\rho_{light}+(\rho_{heavy}-\rho_{light}) \frac{1}{2}\left( 1+\tanh\left( y-\frac{y_{\text{interface}}}{\delta} \right) \right)$$
	3. **Level Set Field**: Initialize $\phi(x,y)$: $$\phi(x,y)=y-y_{\text{interface}}$$
2. **Time Evolution Loop**
	1. **Advection of** $\phi$: $$\phi_{i,j}^{n+1}=\phi^{n}_{i,j}-\Delta t\left( u_{i,j} \frac{{\partial \phi}}{\partial x}+v_{i,j} \frac{{\partial \phi}}{\partial y} \right)$$
  		- Use Eulerian time stepping
	3. **Reinitialize** $\phi$
		- Solve the reinitialization iteratively to maintain $\phi$ as a signed distance function: $$\frac{{\partial \phi}}{\partial \tau} = sign(\phi_{0})(1-|\nabla \phi|)$$
	4. **Density Update**:  
		- Recompute $\rho$ using $H(\phi)$
	5. **Vorticity Update**:
		- Evolve $\omega$ using the vorticity equation
	6. **Solve Poisson Equation**: 
		- Use numerical solvers (e.g. Successive Over-Relaxation or Conjugate Gradient) to compute $\psi$ from $\nabla^{2}\psi=-\omega$
	7. **Update Velocities**
		- Compute $u$ and $v$ from $\psi$
	8. **Visualization**:
		- Plot:
			- The zero-contour of $\phi$ (the interface)
			- Vorticity or density fields perhaps as background color maps
