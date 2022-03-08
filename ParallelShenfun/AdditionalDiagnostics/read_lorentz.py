#!/usr/bin/env python

# post-processing script to plot the lorentz force from the
# output of the qgmhd_shenfun.py driver script.
#
# Options to make a movie or just animate

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from mpi4py import MPI
from shenfun import *
from library.operators import inner, cross
from library.data_output import merge_to_mp4
import subprocess

# mpi imports
comm = MPI.COMM_WORLD 
rank = comm.Get_rank()

# resolution to pick the correct files
N = [1024]*2

# options to make the movie
movie = True #False
movie_name = 'qgmhd_movie.mp4'

# file selection
folder = 'output-qgmhd'
file_name = folder + "/qgmhd_Nx{}_diagnostics.h5".format(N[0])
print('Read from file', file_name)

# read domain variables
file2  = h5py.File(file_name, "r")
domain = file2['domain']
assert(N[0] == file2['size'][0]), "Number of X domain points doesn't match!"
assert(N[1] == file2['size'][1]), "Number of Y domain points doesn't match!"

# read physical parameters
F2     = file2['F2'][...]
M2     = file2['M2'][...]
Uj     = file2['Uj'][...]
B0     = file2['B0'][...]
MHD    = file2['MHD'][...]
Re     = file2['Re'][...]
Rm     = file2['Rm'][...]

# temporal parameters
times  = file2['times']
twrite = file2['twrite']
t0,tf,dt = times[0], times[1], times[2]
tt       = np.arange(t0, tf, dt)
ntplot   = int(twrite[...]/times[2])
tplot    = tt[0::ntplot]

# Display extracted physical parameters
print(' ')
print('Parameters')
print('==========')
print('F2 = ', F2)
print('M2 = ', M2)
print('Uj = ', Uj)
print('B0 = ', B0)
print('N  = (%d,%d) ' % (N[0],N[1]))
print(' ')


# read in the file with the fields
file_name = folder + "/qgmhd_Nx{}_fields.h5".format(N[0])
print('Read from file', file_name)

# Get information from file
f = h5py.File(file_name, 'r')                # open file
keys = list(f.keys())                        # list variables
var1 = keys[0]
x0 = np.array(f[var1 + '/domain/x0'][()])       # bounds in 0-axis
x1 = np.array(f[var1 + '/domain/x1'][()])       # bounds in 1-axis
L = (x0[-1], x1[-1])
timeindex = np.array(f[var1 + '/2D'])           # indices in time
timeindex = np.sort(timeindex.astype(int))
Nt = len(timeindex)

# Create necessary function spaces and spectral variables
V1  = FunctionSpace(N[0], 'F', dtype='D', domain=(0, L[0]))
V2  = FunctionSpace(N[1], 'F', dtype='d', domain=(0, L[1]))
T   = TensorProductSpace(MPI.COMM_SELF, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
TV  = VectorSpace(T)
X   = T.local_mesh(True)
K   = T.local_wavenumbers(True,True)

gradj = Array(TV)

cnt = 0
plt.figure(figsize=(8,6))

b_bar = np.array(f['b_bar/2D/0'][()])

for ii in np.arange(0, Nt):

    j = np.array(f['j/2D/' + str(ii)][()])  
    b = np.array(f['b/2D/' + str(ii)][()])

    b_total = b + b_bar

    for i in range(2):
        gradj[i][:] = T.backward(1j*K[i]*T.forward(j))   

    t = tplot[ii]
    
    plt.clf()
    plt.pcolormesh(X[0], X[1], M2*inner(b,gradj), shading = 'gouraud', cmap='seismic')
    plt.title(r'Lorentz Force $M^2 b\cdot\nabla j$ at t = %6.2f' % t)
    plt.colorbar()
    plt.clim([-np.max(abs(M2*inner(b,gradj))),np.max(abs(M2*inner(b,gradj)))])
    plt.draw()
    plt.pause(0.001)
    
    cnt += 1
    
    if movie:
        plt.savefig('frame_{0:04d}.png'.format(ii), dpi=200)
        
if movie:
    merge_to_mp4('frame_%04d.png', movie_name)




