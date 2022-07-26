# Quasi-Geostrophic Magnetohydrodynamics

This repository contains codes to study Quasi-Geostrophic Magnetohydrodynamics (QG-MHD). 

`ParallelShenfun` solves the QG-MHD equations formulated for the potential vorticity $`q`$, and magnetic streamfunction $`A`$, by substracting a stationary background state $`(\bar q ,\bar A)`$ from the fields and evolving their doubly-periodic pertubations $`(q' ,A')`$, where $`q = \bar q + q'`$, $`A = \bar A + A'`$. 
The code is written in Python and runs in parallel using [Shenfun](https://shenfun.readthedocs.io/en/latest/index.html).

The nonlinear equations read:

   $`\partial_t q + {\bf u}  \cdot {\bf \nabla} q =  M^2 {\bf b} \cdot {\bf \nabla}  \left( \nabla^2 A \right) + \frac{1}{R_e}\nabla^2q,`$

   $`\partial _t A + {\bf u} \cdot {\bf \nabla} A  =  \frac{1}{R_m}\nabla^2A,`$

where (for a kinetic streamfunction $`\psi`$, and magnetic streamfunction $`A`$)

   $`q  = \nabla^2 \psi - F^2 \psi, \quad {\bf u}  = \hat{z}\cdot\nabla\times \psi, \quad {\bf b}  = \hat{z}\cdot\nabla\times A.`$
  
The nondimensional parameters are 

   $`F = \frac{L}{R_d}, \quad M = \frac{V_A}{U}, \quad R_e = \frac{\nu}{UL}, \quad R_m = \frac{\kappa}{UL}`$
   
where $`R_d=\sqrt{gh}/f_0,`$ and $`V_A=B_0/\sqrt{\mu\rho}`$ are the external Rossby radius of deformation and the Aflvèn wave speed, respectively, for gravity ($`g`$), depth ($`h`$), Coriolis frequency $`f_0`$, magnetic field strength ($`B_0`$), magnetic permeability ($`\mu`$) and fluid density ($`\rho`$). $`R_e`$ and $`R_m`$ are the hydrodynamic and magnetic Reynolds numbers with viscosity ($`\nu`$) and magnetic diffusivity ($`\kappa`$). 

For the original derivation or the dimensional equations, see: [Zeitlin, V. (2013). Remarks on rotating shallow-water magnetohydrodynamics.](https://www.semanticscholar.org/paper/Remarks-on-rotating-shallow-water-Zeitlin/b2b294b16feaafecc4b17926d0128894c8153860)

For no magnetic field, set $`M=0`$. For no free-surface, set $`F=0`$.

Perturbations are evolved on a doubly periodic rectangle using a pseudo-spectral method with a smooth filter and AB3 timestepping.

Example plot: Vorticity snapshot of an unstable Bickley jet (without any magnetism) in the nonlinear regime. 

<img src="Images/jet.png" alt="" width="400" height="400"/>

Example Animation: Vorticity of 2D MHD turbulence

<img src="Images/mhd-pv.mp4" alt="" width="500" height="500"/>

`Shenfun` Installation: Please see the relevant instructions at [shenfun.readthedocs.io](https://shenfun.readthedocs.io/en/latest/installation.html)

Other Requirements: `numpy`, `scipy`, `pyfftw`, `matplotlib`, `h5py`, `netCDF4`, `mpi4py`, `sys`, `subprocess`, `os`, `time`.

*For `shenfun` install issues on Compute Canada clusters, try:
```
module load python/3.8.10
module load mpi4py
module load fftw-mpi/3.3.8
virtualenv --no-download ENVgit
source ENVgit/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy scipy pyfftw cython pyyaml sympy mpi4py_fft numba
git clone https://github.com/spectralDNS/shenfun.git
cd shenfun
python setup.py build install
```
