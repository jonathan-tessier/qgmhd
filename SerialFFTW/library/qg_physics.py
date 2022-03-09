#!/usr/bin/env python
# qgmhd_1L_pyfft.py
#
# Evolve the magnetic induction equation with advection and diffusion
#
# Fields: 
#   p  : streamfunction
#   q  : Potential Vorticity
#   A  : Magnetic streamfunction
#   ux : zonal velocity
#   uy : meridional velocity
#   bx : zonal magnetism
#   by : meridional magnetism
#
# Evolution Eqns:
#       q_t = - ux q_x - uy q_x + M2(bx j_x + by j_y) + 1/Re*L*q
#	A_t = - ux A_x - uy A_y                       + 1/Rm*L*A
#       q   = L*p - F2*p
#       r   = L*A
#       u   = (-p_y, p_x)
#       b   = (-A_y, A_x)

import numpy as np
from scipy.fftpack import fft, ifft, fftn, ifftn

from library.operators import inner, jacobian

# spectral stuff
try:
    import pyfftw
    from numpy import zeros as nzeros

    # Keep fft objects in cache for efficiency
    nthreads = 1
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1e8)
    def empty(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

    def zeros(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align(nzeros(N, dtype=dtype), bytes)
    
    # Monkey patches for fft
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    fftn = pyfftw.interfaces.numpy_fft.fftn

except:
    print(" ")
    print(Warning("Install pyfftw, it is much faster than numpy fft"))
    print(" ")
    from numpy.fft import fftn, ifftn

# Set up params stuff
class physical_parameters(object):

    def __init__(
            self,
            
            # Default parameters
            
            F2    = 0.0,                # Froude parameter
            M2    = 1e-4,               # Magnetism parameter
            Re    = 1e16,               # Reynolds parameter
            Rm    = 1e16,               # Magnetic diffusion parameter
            state = 'jet_bickley',      # Initial structure
            Uj    = 1.0,                # Maximum speed
            amp   = 1e-6                # amplitude of initial perturbation

    ):
        
        """
        Physical Parameters
        """
            
        self.F2    = F2
        self.M2    = M2
        self.Re    = Re
        self.Rm    = Rm
        self.state = state
        self.Uj    = Uj
        self.amp   = amp

        """
        Diagnostic Fields
        """
        
        self.diag_fields = ['KE', 'PE', \
                            'q2', 'Ave q', 'grow_q', \
                            'ME', \
                            'A2', 'Ave A', \
                            'Ave j', 'grow_A', 'Helicity']

        self.Nd = len(self.diag_fields)

# diagnostics stuff
def output_diagnostics(diagvals, cnt, t):

    TE    = diagvals[cnt,0] + diagvals[cnt,1] + diagvals[cnt,5]
    print("time=%3.3f, Energy=%10.10f, A2=%10.10f, Helicity=%10.10f, Pert q=%10.10e, Pert A=%10.10e"
          % (t, TE,  diagvals[cnt,6], diagvals[cnt,10], diagvals[cnt,4], diagvals[cnt,9]))

# Spectral flux stuff
def flux_qgmhd(soln, soln_bar, phys):

    Ny,Nx = soln.grid.Ny, soln.grid.Nx
    Re,Rm = phys.Re, phys.Rm
    M2    = phys.M2
    F2    = phys.F2
    
    # Compute scalars
    q_hat    = fftn(soln.q)
    Lapq     = (ifftn(-soln.grid.K2*q_hat)).real
    soln.psi = (ifftn(-soln.grid.K2F2inv*q_hat)).real
    A_hat = fftn(soln.A)
    j_hat = -soln.grid.K2*A_hat
    soln.j     = (ifftn( j_hat )).real

    # Compute vectors
    for i in range(2):
        
        ip = (i + 1) % 2
        
        soln.u[i]     = (ifftn( (-1)**i*soln.grid.iKoK2F2[ip]*q_hat)).real + soln_bar.u[i]
        soln.b[i]     = (ifftn(-(-1)**i*soln.grid.iK[ip]*A_hat)).real      + soln_bar.b[i]
        soln.gradq[i] = (ifftn( soln.grid.iK[i]*q_hat)).real     + soln_bar.gradq[i]
        soln.gradj[i] = (ifftn( soln.grid.iK[i]*j_hat)).real     + soln_bar.gradj[i]
     
    # Compute fluxes
    # FJP: how add on dissiaption and magnetism?
    flux    =   np.zeros((2,Ny,Nx))
    flux[0] = - inner(soln.u, soln.gradq) + M2*inner(soln.b, soln.gradj) + soln.fq + 1/Re*Lapq;
    flux[1] = - jacobian(soln.u, soln.b)                                 + soln.fA + 1/Rm*soln.j;

    q_total = soln.q + soln_bar.q
    A_total = soln.A + soln_bar.A
    j_total = soln.j + soln_bar.j

    Area_xy = soln.grid.Lx*soln.grid.Ly

    # Compute diagnostics
    norms  = np.array((
        np.mean(0.5*inner(soln.u,soln.u))*Area_xy,
        np.mean(0.5*F2*soln.psi**2)*Area_xy,
        np.mean(q_total**2)*Area_xy,
        np.mean(q_total)*Area_xy,
        np.linalg.norm(soln.q)/Nx,
        np.mean(0.5*M2*inner(soln.b,soln.b))*Area_xy,
        np.mean(A_total**2)*Area_xy,
        np.mean(A_total)*Area_xy,
        np.linalg.norm(soln.j)/Nx,
        np.linalg.norm(soln.A)/Nx,
        np.mean(inner(soln.u,soln.b))*Area_xy
    ))

    return flux, norms

# filter stuff
def spectral_filter(soln):
    soln.q  = (ifftn(soln.grid.sfilt*fftn(soln.q))).real
    soln.A  = (ifftn(soln.grid.sfilt*fftn(soln.A))).real

