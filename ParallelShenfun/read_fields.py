#!/usr/bin/env python

# post-processing script to plot the fields from the
# output of the qgmhd_shenfun.py driver script.
#
# Current outputs: q,j,A,psi
#
# Options to make a movie or just animate

#imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from mpi4py import MPI
from shenfun import *
import subprocess
from library.data_output import merge_to_mp4

# mpi imports
comm = MPI.COMM_WORLD 
rank = comm.Get_rank()

# resolution to pick the correct files
N = [1024]*2

# options to make the movie
movie = False #False
movie_name = 'qgmhd_movie.mp4'

# plot full field 
full_field = False

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

# Create Function Spaces
V1  = FunctionSpace(N[0], 'F', dtype='D', domain=(0, L[0]))
V2  = FunctionSpace(N[1], 'F', dtype='d', domain=(0, L[1]))
T   = TensorProductSpace(comm, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
X  = T.local_mesh(True)

cnt = 0
plt.figure(figsize=(9,8))

for ii in np.arange(0, Nt):

    q = np.array(f['q/2D/' + str(ii)][()])  
    j = np.array(f['j/2D/' + str(ii)][()])
    A = np.array(f['A/2D/' + str(ii)][()])
    p = np.array(f['psi/2D/' + str(ii)][()])
    #u = np.array(f['u/2D/' + str(ii)][()])
    #b = np.array(f['b/2D/' + str(ii)][()])

    if full_field:
        q += np.array(f['q_bar/2D/' + str(0)][()])
        A += np.array(f['A_bar/2D/' + str(0)][()])
        p += np.array(f['psi_bar/2D/' + str(0)][()])
        #u += np.array(f['u_bar/2D/' + str(0)][()])
        #b += np.array(f['b_bar/2D/' + str(0)][()])

    t = tplot[ii]
    if M2==0: M2=1;
     
    plt.clf()
    plt.subplot(2,2,1)
    plt.pcolormesh(X[0], X[1], q, cmap = 'seismic', shading = 'gouraud')
    plt.title('q at t = %6.2f' % t)
    plt.clim([-np.max(abs(q)),np.max(abs(q))])
    plt.colorbar()
    plt.ylabel("y"); #plt.xlabel("x")

    plt.subplot(2,2,2)
    plt.pcolormesh(X[0], X[1], np.sqrt(M2)*j, cmap = 'seismic', shading = 'gouraud')
    plt.title('jM at t = %6.2f' % t)
    plt.clim([-np.max(abs(np.sqrt(M2)*j)),np.max(abs(np.sqrt(M2)*j))])
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.pcolormesh(X[0], X[1], p, cmap = 'seismic', shading = 'gouraud')
    plt.title('psi at t = %6.2f' % t)
    plt.clim([-np.max(abs(p)),np.max(abs(p))])
    plt.colorbar()
    plt.ylabel("y"); plt.xlabel("x")

    plt.subplot(2,2,4)
    plt.pcolormesh(X[0], X[1], np.sqrt(M2)*A, cmap = 'seismic', shading = 'gouraud')
    plt.title('AM at t = %6.2f' % t)
    plt.clim([-np.max(abs(np.sqrt(M2)*A)),np.max(abs(np.sqrt(M2)*A))])
    plt.colorbar()
    plt.xlabel("x")
    
    plt.draw()
    plt.pause(0.001)
    
    cnt += 1

    if movie:
        plt.savefig('frame_{0:04d}.png'.format(ii), dpi=200)
        
if movie:
    merge_to_mp4('frame_%04d.png', movie_name)




