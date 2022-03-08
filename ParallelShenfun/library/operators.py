#!/usr/bin/env python

# src script used by the qgmhd_shenfun.py script to define basic
# functions and the backgroudn states used in evolution.

# Add additional background states here as needed and import them in
# the driver script (qhmhd_shenfun.py) under the library imports.
# Undefined background variables are zero by default.

# basic imports
import numpy as np

# Basic inner product of vectors
def inner(u, v):
    return u[0]*v[0] + u[1]*v[1]

# Basic Jacobian of vectors
def cross(u, v):
    return u[0]*v[1] - u[1]*v[0]
    
# background state of a Bickley jet in constant zonal magnetic field
def jet_bickley(phys, grid, soln):
    soln.qA[0]    =  2.*phys.Uj*np.tanh(grid.X[1] - grid.Ly/2.)/pow(np.cosh(grid.X[1] - grid.Ly/2.),2) \
                     + phys.F2*phys.Uj*np.tanh(grid.X[1] - grid.Ly/2.)
    soln.qA[1]    = - phys.B0*grid.X[1]
    soln.psi[:]   = - phys.Uj*np.tanh(grid.X[1] - grid.Ly/2.)
    soln.u[0]     = phys.Uj/pow(np.cosh(grid.X[1] - grid.Ly/2),2)
    soln.gradq[1] = 4.*phys.Uj*(0.5 - (np.sinh(grid.X[1] - grid.Ly/2)**2))/(np.cosh(grid.X[1] - grid.Ly/2))**4 \
                    + phys.F2*phys.Uj/pow(np.cosh(grid.X[1] - grid.Ly/2.),2)
    soln.b[0]     = 0*grid.X[1] + phys.B0
