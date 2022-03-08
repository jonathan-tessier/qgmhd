#!/usr/bin/env python
#
# src script for qgmhd_shenfun.py. Defines the physical parameters
# and evolve the PV and the magnetic induction equation.
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
#       q_t = - ux q_x - uy q_x + M2(bx j_x + by j_y) + 1/Re*Lap(q)
#	A_t = - ux A_x - uy A_y                       + 1/Rm*Lap(A)
#       q   = Lap(p) - F2*p
#       j   = Lap(A)
#       u   = (-p_y, p_x)
#       b   = (-A_y, A_x)

import numpy as np
from shenfun import *
from library.operators import inner, cross

# Set up parameters
class physical_parameters(object):

    def __init__(
            self,
            
            # Default parameters (can be editted via driver script: qgmhd_shenfun.py)

            # switches
            MHD     = True,          # switch (turns MHD/Lorentz-Force on/off)
            flux_qg = True,          # turns on PV equation
            flux_mhd= True,          # turns on A quation
            fluxRe  = False,         # PV viscosity
            fluxRm  = False,         # A diffusion
            fluxFq  = False,         # PV forcing
            fluxFA  = False,         # A forcing

            # Physical
            F2      = 0.0,           # Froude parameter
            M2      = 1e-4,          # Magnetism parameter
            Re      = 1e16,          # Reynolds parameter
            Rm      = 1e16,          # Magnetic diffusion parameter
            state   = 'jet_bickley', # Initial structure
            Uj      = 1.0,           # Maximum speed
            B0      = 1.0,           # background zonal field amp

            # perturbations
          psi_amp   = 1e-2,          # amplitude of initial psi perturbation
            A_amp   = 1e-2,          # amplitude of initial A perturbation
          psi_sig   = 2,             # spectral width of initial q perturbation
            A_sig   = 2,             # spectral width of initial A perturbation
          psi_nps   = 9,             # np.seed of initial q perturbation
            A_nps   = 11,            # np.seed of initial A perturbation
          psikstar  = 0.5,           # central wavenumber for initial q pert spectrum
            Akstar  = 0.5,           # central wavenumber for initial A pert spectrum

            # forcing
            Fq_amp  = 0,             # amplitude of q forcing
            FA_amp  = 0,             # amplitude of A forcing
            Fq_sig  = 1,             # spectral width of q forcing
            FA_sig  = 1,             # spectral width of A forcing
            Fq_nps  = 0,             # np.seed of q forcing
            FA_nps  = 1,             # np.seed of A forcing
            Fqkstar = 5,             # central wavenumber for q forcing spectrum
            FAkstar = 3,             # central wavenumber for A forcing spectrum

            display = True           # show physical parameters in terminal out.
    ):
        
        """
        Physical Parameters
        """
            
        self.F2      = F2
        self.M2      = M2
        self.Re      = Re
        self.Rm      = Rm
        self.state   = state
        self.Uj      = Uj

        self.psi_amp = psi_amp
        self.A_amp   = A_amp
        self.psi_sig = psi_sig
        self.A_sig   = A_sig
        self.psi_nps = psi_nps
        self.A_nps   = A_nps
        self.psikstar= psikstar
        self.Akstar  = Akstar

        self.Fq_amp  = Fq_amp
        self.FA_amp  = FA_amp
        self.Fq_sig  = Fq_sig
        self.FA_sig  = FA_sig
        self.Fq_nps  = Fq_nps
        self.FA_nps  = FA_nps
        self.Fqkstar = Fqkstar
        self.FAkstar = FAkstar

        self.B0      = B0
        self.MHD     = MHD

        self.flux_qg = flux_qg
        self.flux_mhd= flux_mhd
        self.fluxRe  = fluxRe
        self.fluxRm  = fluxRm
        self.fluxFq  = fluxFq
        self.fluxFA  = fluxFA

        self.display = display

        """
        Diagnostic Fields
        """
        
        self.diag_fields = ['KE', 'PE', \
                            'q2', 'Ave q', 'grow_q', \
                            'ME', \
                            'A2', 'Ave A', \
                            'Ave j', 'grow_A', 'Helicity', \
                            'u2', 'v2', 'b12','b22','qx2','qy2','jx2','jy2','lorentz']

        self.Nd = len(self.diag_fields)

        if self.display:
            print(' ')
            print('Physical Parameters')
            print('===================')
            print('F2    = ', self.F2, '\nM2    = ', self.M2, \
                  '\nUj    = ', self.Uj, '\npsi_amp = ', self.psi_amp, \
                  '\nA_amp  = ', self.A_amp, '\nstate = ', self.state)


# Terminal Output: diagnostics
def output_diagnostics(diagvals, cnt, t):

    TE    = diagvals[cnt,0] + diagvals[cnt,1] + diagvals[cnt,5]
    print("time=%3.3f, Energy=%10.10f, A2=%10.10f, Helicity=%10.10f, Pert q=%10.10e, Pert A=%10.10e"
          % (t, TE,  diagvals[cnt,6], diagvals[cnt,10], diagvals[cnt,4], diagvals[cnt,9]))

# Computes/Completes the Initial condtions from only q and A
def compute_ICs_from_qA(soln, soln_bar, phys, T, TV, VM):

    # Compute scalars
    soln.psi[:] = T.backward(-soln.grid.K2F2inv*soln.qA_hat[0], soln.psi)
    soln.j[:]   = T.backward(-soln.grid.K2*soln.qA_hat[1], soln.j)

    # Compute vectors
    for i in range(2):
        ip = (i + 1) % 2
        soln.u[i][:]     = T.backward( (-1)**i*soln.grid.iKoK2F2[ip]*soln.qA_hat[0], soln.u[i])
        soln.b[i][:]     = T.backward(-(-1)**i*soln.grid.iK[ip]*soln.qA_hat[1],    soln.b[i])
    soln.u     += soln_bar.u
    soln.b     += soln_bar.b

# flux computation for PV and A evolution
def flux_qgmhd(soln, soln_bar, phys, T, TV, VM):

    Ny,Nx = soln.grid.Ny, soln.grid.Nx
    Re,Rm = phys.Re, phys.Rm
    M2    = phys.M2
    F2    = phys.F2
    Re    = phys.Re
    Rm    = phys.Rm
    MHD   = phys.MHD

    # Compute scalars
    soln.psi[:] = T.backward(-soln.grid.K2F2inv*soln.qA_hat[0], soln.psi)
    j_hat = -soln.grid.K2*soln.qA_hat[1]
    soln.j[:] = T.backward( j_hat, soln.j)

    # Compute vectors
    for i in range(2):
        
        ip = (i + 1) % 2
        soln.u[i][:]     = T.backward( (-1)**i*soln.grid.iKoK2F2[ip]*soln.qA_hat[0], soln.u[i])
        soln.b[i][:]     = T.backward(-(-1)**i*soln.grid.iK[ip]*soln.qA_hat[1],    soln.b[i])    
        soln.gradq[i][:] = T.backward( soln.grid.iK[i]*soln.qA_hat[0],             soln.gradq[i])
        soln.gradj[i][:] = T.backward( soln.grid.iK[i]*j_hat,                      soln.gradj[i])

    # add background contributions to velocity and magnetic field.
    soln.u     += soln_bar.u
    soln.b     += soln_bar.b
    soln.gradq += soln_bar.gradq
    soln.gradj += soln_bar.gradj

    # Compute fluxes:
    flux = Array(VM) 

    # evolve the PV equation
    if phys.flux_qg:
        flux[0][:] += - inner(soln.u, soln.gradq) + MHD*M2*inner(soln.b, soln.gradj)

    # evolve the A equation
    if phys.flux_mhd:
        flux[1] += - cross(soln.u, soln.b)

    # add viscosity (PV eq.)
    if phys.fluxRe:
        q_hat = soln.qA_hat[0]
        Lapq     = Array(T)
        Lapq     = T.backward(-soln.grid.K2*q_hat, Lapq)
        flux[0] += 1/Re*Lapq

    # add diffusion (A eq.)
    if phys.fluxRm:
        flux[1] += 1/Rm*soln.j

    # add kinetic forcing (PV eq.)
    if phys.fluxFq:
        flux[0] += soln.FqA[0]

    # add magnetic forcing (A eq.)
    if phys.fluxFA:
        flux[1] += soln.FqA[1]

    # compute the total q and A for diagnostics
    qA_total = soln.qA + soln_bar.qA
    # domain area for total field computations
    Area_xy = soln.grid.Lx*soln.grid.Ly

    # Compute diagnostics
    norms  = np.array((
        np.mean(0.5*inner(soln.u,soln.u))*Area_xy,
        np.mean(0.5*F2*(soln.psi+soln_bar.psi)**2)*Area_xy,
        np.mean(qA_total[0]**2)*Area_xy,
        np.mean(qA_total[0])*Area_xy,
        np.linalg.norm(soln.qA[0])/Nx,
        np.mean(0.5*MHD*M2*inner(soln.b,soln.b))*Area_xy,
        np.mean(M2*qA_total[1]**2)*Area_xy,
        np.mean(np.sqrt(M2)*qA_total[1])*Area_xy,
        np.mean(M2*soln.j**2)*Area_xy,
        np.linalg.norm(np.sqrt(M2)*soln.qA[1])/Nx,
        np.mean(inner(soln.u,np.sqrt(M2)*soln.b))*Area_xy,
        np.mean(soln.u[0]**2)*Area_xy,
        np.mean(soln.u[1]**2)*Area_xy,
        np.mean(M2*soln.b[0]**2)*Area_xy,
        np.mean(M2*soln.b[1]**2)*Area_xy,
        np.mean(soln.gradq[0]**2)*Area_xy,
        np.mean(soln.gradq[1]**2)*Area_xy,
        np.mean(M2*soln.gradj[0]**2)*Area_xy,
        np.mean(M2*soln.gradj[1]**2)*Area_xy,
        np.mean(M2*inner(soln.b,soln.gradj))*Area_xy
    ))
    return flux, norms

# filter function
def spectral_filter(soln, T, VM):
    soln.qA_flux = VM.forward(soln.qA, soln.qA_hat)
    for i in range(2):
        soln.qA[i][:]  = T.backward( soln.grid.sfilt*soln.qA_flux[i], soln.qA[i])


