#!/usr/bin/env python

#####################################################
# Driver script to consider an unstable Bickley Jet #
#####################################################

import numpy as np
import time

# Import stuff from this library
from library.operators     import jet_bickley
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

    phys  = physical_parameters(F2 = 0.0, M2 = 0.1, Uj = 1.0, Re = 1e4, Rm = 1e4, amp = 1e-4)

    ## Temporal Parameters:
    # tf: Final time
    # dt: timstep
    # tplot: plotting frequency

    times = temporal_parameters(tf = 200.0, dt = 1e-2, tplot = 1.0)

    ## grid parameters:
    # M: resolution power (Nx = Ny = 2^M) M = 7 => 128x128 grid
    # Lx/Ly: domain lengths

    grid  = grid_parameters(phys, M = 7, Lx = 20, Ly = 20)

    # Output parameters to IO
    if display:
        display_parameters(phys, times, grid)                                     

    # output data folder location
    folder = 'output-qgmhd'
    gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2, diagvals = initialize_output(phys, times, grid, folder) 

    # data initialization
    soln, soln_bar = init_solution(grid), init_solution(grid)
 
    # Set background state (Can create additional states in library/operators.py)
    jet_bickley(phys, grid, soln_bar)

    # Set perturbations: uses an annulus or bump in spectral space to create a random field as the perturbation.
    soln.q  = generate_broad_spectrum_field(grid = grid, kstar = 10, sigma = 0.5, npseed = 0, amp=phys.amp)
    #soln.A  = generate_field_with_kpeak(grid = grid, kstar = 10, sigma = 0.5, npseed = 1, amp=phys.amp)

    # For a smooth/Gaussian in Fourier space, use: 
    # generate_broad_spectrum_field(grid = grid, kstar = <characteristic wavenumber>, sigma = <width>, npseed = <random seed>, amp=phys.amp)
    # For a sharp annulus at a specific wavenumber, use:
    # generate_field_with_kpeak(grid = grid, kstar = <characteristic wavenumber>, sigma = <width>, npseed = <random seed>, amp=phys.amp)
    
    # set time and solve
    time_initial = time.time()
    solve_model(phys, times, soln, soln_bar, gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2, diagvals)

    print('\nTotal time = ', (time.time() - time_initial))
        
main()

