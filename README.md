# Nonlinear Magnetohydrodynamics

This repository contains a variety of codes to study Magnetohydrodynamics. 

The current code solves the rotating, incompressible, shallow MHD equations formulated for the potential vorticity $`q`$, and magnetic streamfunction $`A`$:

   $`\partial_t q + {\bf u}  \cdot {\bf \nabla} q =  M^2 {\bf b} \cdot {\bf \nabla}  \left( \nabla^2 A \right) + \frac{1}{R_e}\nabla^2q,`$

   $`\partial _t A + {\bf u} \cdot {\bf \nabla} A  =  \frac{1}{R_m}\nabla^2A,`$

where (for a velocity streamfunction $`\psi`$, and magnetic streamfunction $`A`$)

   $`q  = \nabla^2 \psi - F^2 \psi, \quad {\bf u}  = \hat{z}\cdot\nabla\times \psi, \quad {\bf b}  = \hat{z}\cdot\nabla\times A.`$
  
The nondimensional parameters are 

   $`F = \frac{L}{R_d}, \quad M = \frac{V}{U}, \quad R_e = \frac{\nu}{UL}, \quad R_m = \frac{\kappa}{UL}`$
   
where $`R_d, V, \nu, \kappa`$ are the external Rossby radius of deformation, the Aflvèn wave speed, the viscosity and magnetic diffusivity, respectively. 

For the original derivation or the dimensional equations, see: 

Zeitlin, V. (2013). Remarks on rotating shallow-water magnetohydrodynamics. Nonlinear Processes in Geophysics, 20, 893-898.


For no influence of the coriolis force, set $`F=0`$. For no influence of the magnetic field, set $`M=0`$. 

Equations are solved using a spectral method (FFTW) with a smooth filter and AB3 timestepping. 

