#!/usr/bin/env python

# src script used by the qgmhd_shenfun.py script to solve/evolve
# the PV and A equations. Defines the time-stepping scheme and
# filters the equations.

# imports
from library.qg_physics import flux_qgmhd, spectral_filter, output_diagnostics, compute_ICs_from_qA
from library.data_output import plot_field
import numpy as np

# Set up params stuff        
class temporal_parameters(object):

    def __init__(
            self,
            t0      = 0.0,   # initial time
            tf      = 1.0,   # final time
            dt      = 1e-2,  # timestep
            tplot   = 1.0,   # plotting frequency
            method  = 'AB3', # timestepping method (fixed)
            display = True,  # display timestepping parameters
            onthefly= False  # plot the fields as the simulation is running (avoid in parallel)
    ):

        """
        Temporal Parameters 
        """ 

        self.t0      = t0
        self.tf      = tf
        self.dt      = dt
        self.tplot   = tplot
        self.method  = method
        
        self.t_domain = (self.t0, self.tf+self.dt)

        self.Nt      = int(self.tf/self.dt) + 2
        self.npt     = int(self.tplot/dt)
        self.tt      = np.arange(self.Nt)*dt
        self.ttplt   = np.arange(int((self.Nt-1)/self.npt)+1)*dt*self.npt

        self.display = display
        self.onthefly= onthefly

        if self.display:
            print(' ')
            print('Temporal Parameters')
            print('===================')
            print('t0  = ', self.t0, '\ntf  = ', self.tf, '\ndt  = ', self.dt, \
                  '\nNt  = ', self.Nt)

# time stepping solving
def solve_model(phys, times, soln, soln_bar, file, diagvals, T, TV, VM):

    # Initialize fields
    Nx, Ny = soln.grid.Nx, soln.grid.Ny

    ### Euler step
    cnt = 0
    t   = times.t0 + cnt*times.dt
    # from q and A compute psi, j, u and b
    compute_ICs_from_qA(soln, soln_bar, phys, T, TV, VM)

    # write the initial conditions and background fields to file
    if cnt % times.npt == 0:
        cnt_nc = int(cnt/times.npt)
        file.write(0, {'b' : [ soln.b-soln_bar.b ], 'u' : [ soln.u-soln_bar.u ] ,\
                       'psi' : [ soln.psi ], 'q' : [ soln.qA[0] ], 'A' : [ soln.qA[1] ], 'j' : [ soln.j ]})
        file.write(0, {'q_bar' : [ soln_bar.qA[0] ], 'psi_bar' : [ soln_bar.psi ], \
                       'A_bar' : [ soln_bar.qA[1] ], 'u_bar' : [ soln_bar.u ], 'b_bar' : [ soln_bar.b ]})
        file.write(0, {'Fq' : [ soln.FqA[0] ], 'FA' : [ soln.FqA[1] ]})
        # plot solution is wanted
        if times.onthefly: plot_field(soln,t);

    # compute the flux to the equations
    NLnm, diagvals[cnt,:] = flux_qgmhd(soln, soln_bar, phys, T, TV, VM)
    # update solution: Forward Euler
    soln.qA = soln.qA + times.dt*NLnm  
    # output diagnostics to terminal
    output_diagnostics(diagvals, cnt, t);
    # filter solution
    spectral_filter(soln, T, VM)

    ### AB2 step
    cnt = 1
    t   = times.t0 + cnt*times.dt
    # write solution to file
    if cnt % times.npt == 0:
        cnt_nc = int(cnt/times.npt)
        file.write(cnt_nc, {'b' : [ soln.b-soln_bar.b ], 'u' : [ soln.u-soln_bar.u ] , \
        'psi' : [ soln.psi ], 'q' : [ soln.qA[0] ], 'A' : [ soln.qA[1] ], 'j' : [ soln.j ]} )
        # plot solution is wanted
        if times.onthefly: plot_field(soln,t);

    # solve the flux
    NLn, diagvals[cnt,:] = flux_qgmhd(soln, soln_bar, phys, T, TV, VM)
    # update solution: AB2
    soln.qA = soln.qA + 0.5*times.dt*(3.*NLn - NLnm)  
    # output to terminal
    output_diagnostics(diagvals, cnt, t);
    # filter
    spectral_filter(soln, T, VM)

    # AB3 step
    for cnt in range(2,times.Nt):
        t   = times.t0 + cnt*times.dt
        # solve flux
        NL, diagvals[cnt,:] = flux_qgmhd(soln, soln_bar, phys, T, TV, VM)
        # update: AB3
        soln.qA  = soln.qA + times.dt/12.*(23*NL - 16.*NLn + 5.*NLnm)
        # output to terminal
        output_diagnostics(diagvals, cnt, t);
        # filter solution
        spectral_filter(soln, T, VM)

        # Reset fluxes
        NLnm = NLn
        NLn  = NL

        # write to file
        if cnt % times.npt == 0:
            cnt_nc = int(cnt/times.npt)
            file.write(cnt_nc, {'b' : [ soln.b-soln_bar.b ], 'u' : [ soln.u-soln_bar.u ] ,\
            'psi' : [ soln.psi ], 'q' : [ soln.qA[0] ], 'A' : [ soln.qA[1] ], 'j' : [ soln.j ]} )
            # plot solution is wanted
            if times.onthefly: plot_field(soln,t);
