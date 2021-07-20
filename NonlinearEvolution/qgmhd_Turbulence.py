#!/usr/bin/env python

#####################################################
# Driver script to consider an 2D MHD turbulence    #
#####################################################

import numpy as np
from scipy.fftpack import fft, ifft, fftn, ifftn
import time

# Import stuff from this library
from library.operators     import jet_bickley, vortex_orszag_tang
from library.time_stepping import temporal_parameters, solve_model
from library.grid          import grid_parameters, generate_field_with_kpeak, generate_broad_spectrum_field
from library.qg_physics    import physical_parameters
from library.display       import display_parameters
from library.initialize    import init_solution
from library.data_output   import build_netcdf, initialize_output, plot_field

display = True

def main():

    ## Physical Parameters:
    # F2:    Corioils
    # M2:    Magnetism
    # Uj:    Background Velocity Amplitude
    # Re/Rm: Reynolds numbers
    # amp:   perturbation amplitude

    phys  = physical_parameters(F2 = 0.25, M2 = 1, Uj = 0.0, Re = 1e4, Rm = 1e4, amp = 1)

    ## Temporal Parameters:
    # tf: Final time
    # dt: timstep
    # tplot: plotting frequency

    times = temporal_parameters(tf = 10, dt = 1e-3, tplot = 0.1)

    ## grid parameters:
    # M: resolution power (Nx = Ny = 2^M) M = 7 => 128x128 grid
    # Lx/Ly: domain lengths

    grid  = grid_parameters(phys, M = 7, Lx = 2*np.pi, Ly = 2*np.pi)

    # Output parameters to IO
    if display:
        display_parameters(phys, times, grid)                                     

    # output data folder location
    folder = 'output-qgmhd'
    gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2, diagvals = initialize_output(phys, times, grid, folder) 

    # data initialization
    soln, soln_bar = init_solution(grid), init_solution(grid) 
   
    # keep commented for self-interaction, uncomment for background magnetic field 
    #jet_bickley(phys, grid, soln_bar) # keeps the constant zonal mag field, Uj=0 means no jet

    # can set the initial condition from the streamfunction and transform to the PV
    psi = generate_broad_spectrum_field(grid = grid, kstar = 5, sigma = 2, npseed = 0, amp=phys.amp)
    soln.q  = ifftn(-grid.K2F2*fftn(psi)).real
    soln.A  = generate_broad_spectrum_field(grid = grid, kstar = 5, sigma = 2, npseed = 1, amp=phys.amp)

    # This adds a forcing term in either evolution equations, currently a random field
    #soln.fq = generate_field_with_kpeak(grid = grid, kstar = 10, sigma = 0.5, npseed = 2, amp=0.1)
    #soln.fA = generate_field_with_kpeak(grid = grid, kstar = 10, sigma = 0.5, npseed = 3, amp=0.1)

    # set time and solve
    time_initial = time.time() 
    solve_model(phys, times, soln, soln_bar, gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2, diagvals)

    print('\nTotal time = ', (time.time() - time_initial))
        
main()

