#!/usr/bin/env python

# Driver Script for QGMHD Simulations using the shenfun spectral Galerkin method:
# - Uses src scripts in /library to evolve doubly periodic perturbations of PV (q)
#   and magnetic streamfunction (A) on top of an optional steady background state.
# - The output of the model is sent to the folder 'output-qgmhd' in h5 format.
# - The output fields and diagnostics are viewable via the read_fields.py and 
#   read_diagnostics.py scripts. For more advanced plotting, see /AdditionalDiagnostics.
# - When running in parallel, use the command $ mpiexec -np X python3 qgmhd_shenfun.py
# - When running on graham (Compute Canada) or other clusters with SLURM, you should use 
#   $ srun  instead of $ mpiexec -np X  for optimal ressource efficiency.
# - At 1024^2 points with dt = 5e-4, tf = 300, tplot = 2, an 8 core parallel simulation takes ~18hrs on graham.
 
# By: Jonathan Tessier, Francis J. Poulin

# General imports
from mpi4py import MPI
import numpy as np
from shenfun import *
import h5py
import time
import sys

# MPI parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 

# Specific library imports
from library.operators     import jet_bickley
from library.qg_physics    import physical_parameters, compute_ICs_from_qA
from library.time_stepping import temporal_parameters, solve_model
from library.grid          import grid_parameters, generate_field_with_kpeak, generate_broad_spectrum_field
from library.initialize    import init_solution, Build_FunctionSpaces, scatter_ICs, scatter_forcing
from library.data_output   import initialize_output, plot_field, generate_diagnostics

def main():
    
    # Domain lengths
    L = [8*np.pi]*2
    # Number of Data Points 
    N = [1024]*2

    # Initilialize the function spaces
    T0, T, TV, VM = Build_FunctionSpaces(N[0], N[1], L[0], L[1])

    # Initialize the Physical Parameters
    phys  = physical_parameters(MHD = True,  # Turns on MHD, Off for pure hydrodynamic flow
                                 F2 = 0,     # Inverse Burger Number L^2/R_d^2
                                 M2 = 1e-4,  # square ratio of Alfven Wave speed to flow velocity
                                 Uj = 0,     # Amplitude of the background Bickley jet
                                 B0 = 1,     # Switch (0,1) to turn on the background mag field
                                 Rm = 1e4,   # Magnetic Reynolds number (Magnetic diffusion)
                                 Re = 1e4,   # Reynolds number (viscosity)
                            psi_amp = 0.2,   # Amplitude of the psi perturbation
                              A_amp = 0.2,   # Amplitude of the A perturbation
                           psikstar = 0,     # Central wavenumber of the psi pert.
                             Akstar = 0,     # Central wavenumber of the A pert.
                            psi_sig = 1,     # Spectral width of the psi pert.
                              A_sig = 1,     # Spectral width of the A pert.
                            psi_nps = 9,     # Numpy seed for random generator psi
                              A_nps = 11,    # Numpy seed for random generator A
                            flux_qg = True,  # Turns on the PV equation, Off for pure induction
                           flux_mhd = True,  # Turns on the A equation, Off for no A evolution
                             fluxRe = True,  # Turns on viscosity
                             fluxRm = True,  # Turns on magnetic diffusion
                             fluxFq = False, # Turns on Forcing in the PV equation
                             fluxFA = False, # Turns on Forcing in the A equation
                             Fq_amp = 0,     # Amplitude of the PV forcing
                             FA_amp = 0,     # Amplitude of the A forcing
                            Fq_sig  = 1,     # spectral width of q forcing
                            FA_sig  = 1,     # spectral width of A forcing
                            Fq_nps  = 0,     # Numpy seed for PV forcing
                            FA_nps  = 1,     # Numpy seed of A forcing
                            Fqkstar = 3,     # Central wavenumber for q forcing spectrum
                            FAkstar = 3)     # Central wavenumber for A forcing spectrum

    # sets timestepping parameters (tf,dt,tplot) = (final time, timestep, plotting frequency)
    # setting onthefly = True makes the code plot q and A as it computes it (use for testing)
    times = temporal_parameters(tf = 150, dt =2.5e-4, tplot = 1, onthefly = False)

    # initializes the grid
    grid  = grid_parameters(phys, T, Nx = N[0], Ny = N[1], Lx = L[0], Ly = L[1])

    # initializes the solution structures (pertubation and background)
    soln, soln_bar = init_solution(grid, T, TV, VM), init_solution(grid, T, TV, VM)
    
    # initializes the output directory and parameters
    folder = 'output-qgmhd' # Data storage folder name
    file0, diagvals = initialize_output(phys, times, grid, folder) 

    # initializes the background field to be a Bickley jet in a constant zonal magnetic field.
    # when Uj = 0, no Bickley jet is present.
    # When B0 = 0, no background magnetic field is present.
    # Turning off flux_qg solves the pure induction problem. 
    # Turning off flux_mhd solves only the PV advection for fixed field.
    # Additional profiles can be added in library/operators.
    jet_bickley(phys, grid, soln_bar)                                         

    # initializes the perturbation fields with ICtype:
    # - broad_random: Random Field, Gaussian in Spectral space, exp(-(K/({q,A}sig*{q,A}kstar))^2)
    # - peak_random: Random Field, Annulus in Spectral space, centered at {q,A}kstar, and of width {q,A}_sig
    # - big_vortex: Large Vortex, half the domain size, centered at (Lx/2,Ly/2)
    # - single_vortex: Smaller vortex, a quarter the domain size, centered at (Lx/2,Ly/2)
    # - quad_vortex: An infinite tile of vortices, half the domain size
    # - row_vortex: A single row of vortices, a quarter the domain size, centered at (y = Ly/2)
    soln.qA[0], soln.psi[:], soln.j[:], soln.qA[1] = scatter_ICs(T0, grid, phys, comm, rank, size, ICtype = 'broad_random')
    
    # Plot the initial fields if needed
    plot_field(soln,0)
    sys.exit()

    # Uncomment to add arbitrary forcing to the equations, currently set-up for forcing at a particular scale
    #soln.FqA[0], soln.FqA[1] = scatter_forcing(T0, grid, phys, comm, rank, size)

    # Create the first instances of the transformed variables
    soln.qA_hat = VM.forward(soln.qA, soln.qA_hat)

    # Simluation Timer
    time_initial = time.time()

    # Start Simulation
    solve_model(phys, times, soln, soln_bar, file0, diagvals, T, TV, VM)

    # Simulation Timer
    print('\nTotal time = ', (time.time() - time_initial))

    # generate field diagnostics
    if rank==0:
        generate_diagnostics(phys,times,grid,folder,T0)

main()
