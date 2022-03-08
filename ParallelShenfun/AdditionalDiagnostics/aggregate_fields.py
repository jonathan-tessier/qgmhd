#!/usr/bin/env python

# post-processing script to read the fields output from 
# multiple runs of the qgmhd_shenfun.py driver script to plot
# snapshots of them as a function of M (mag. field strength)
#
# Current outputs: PV (q), current (j), streamfunction (psi)
# and magnetic streamfunction (A), along with the Lorentz force
#
# In the same directory as this script, create folders named as
# <case_prefix>-M2-<mag_value> Eg: turbulence-M2-1em4, which 
# each contain a 'output-qgmhd' output folder from the model.
# Specify the case_prefix below which is fixed for a set of 
# figure. Further, list the values of magnetism to include,
# so the suffixes to your directory names, in mag_values.
# The code will extract the actual value of M from each output.

# This scripts further requires a symbolic link of the code library.
# from this directory, run $ ln -s /path/to/library
# NOTE: Should avoid needing lib if all we're taking is an inner defn...
#
# As long as nrows*ncols = len(mag_values), you can pick whatever
# configuration of rows and columns you want the panels arranged in.
# The code will go left to right and top to bottom in the order of 
# provided mag_values. This code is however restricted to plotting 
# more than one case. For a single run, consider scripts without 
# the 'aggregate' prefix.

# imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from shenfun import *
from library.operators import inner, cross
from library.data_output import fmt, panelindex
# creating figures for multiple values of M^2.

# set resolution to pick out correct files
N = [1024]*2

# Flag to plot the full fields (q = \bar q + q') or perturbations (q')
full_field = False

# Time to take the snapshot for each field
t_snap = 50

# set the directories you want to read from
case_prefix = "turbB0-F2-0"

# list of directory suffixes corresponding to M2 values 
#mag_values = ["hydro","1em6","1em5","1em4","1em3","1em2","1em1","1em0"]
mag_values = ["hydro","1em6","1em4","1em2"]

# figure panel config
ncols,  nrows  = [int(len(mag_values)), 1]; # rows and cols
hscale, vscale = [2.5,2.5]; # approx size of each panel
figsize=((hscale+0.5)*ncols, vscale*nrows+1) # actual figure size

# check the figure will actually fit all values
assert(ncols*nrows == int(len(mag_values))),"Number of simulations doesn't fit in the grid."

# init dir arrays
diagfilename_array = []
fieldfilename_array = []

# create array of directory files to open
for magval in mag_values:
    diagfilename_array.append(case_prefix+"-M2-"+magval+\
        "/output-qgmhd/qgmhd_Nx{}_diagnostics.h5".format(N[0]))
    fieldfilename_array.append(case_prefix+"-M2-"+magval+\
        "/output-qgmhd/qgmhd_Nx{}_fields.h5".format(N[0]))

# initialize and show all files
print('Read from files \n', fieldfilename_array)
M2_vector = [] # init array of M2 values

# read first diag file for invariant quantities
file0 = h5py.File(diagfilename_array[0], "r")
times  = file0['times']
twrite = file0['twrite']
t0,tf,dt = times[0], times[1], times[2]
tt       = np.arange(t0, tf, dt)
ntplot   = int(twrite[...]/times[2])
tplot    = tt[0::ntplot]
assert(tf>=t_snap),"Snapshot Time Too Large: Outside Simulation Time Range"
ii = int(np.where(np.array(tplot)==t_snap)[0])
domain = file0['domain']
N      = file0['size']
F2     = file0['F2']
Uj     = file0['Uj']

print('Opening diag files to pull M2 values...')

for filename in diagfilename_array:
    filed = h5py.File(filename, "r")
    M2_vector.append(float(filed['M2'][...]))

print(M2_vector)

# first field file for invariant domain quantities
f = h5py.File(fieldfilename_array[0], "r")
keys = list(f.keys())
 
# Which variable to view
var1 = keys[0]

x0 = np.array(f[var1 + '/domain/x0'][()])       # bounds in 0-axis
x1 = np.array(f[var1 + '/domain/x1'][()])       # bounds in 1-axis

print('Lengths of domain:')
print('      x0 in [%6.4f, %6.4f]' % (x0[0], x0[-1]))
print('      x1 in [%6.4f, %6.4f]' % (x1[0], x1[-1]))

L = (x0[-1], x1[-1])

print('Resolutions:')
print('    (Nx,Ny) = (%d, %d)' % (N[0], N[1]))

timeindex = np.array(f[var1 + '/2D'])           # indices in time
timeindex = np.sort(timeindex.astype(int))

Nt = len(timeindex)

print('Outputs in time:')
print('      Nt    = ', Nt)
print('      nplot = ', ntplot)

# Read files
V1  = FunctionSpace(N[0], 'F', dtype='D', domain=(0, L[0]))
V2  = FunctionSpace(N[1], 'F', dtype='d', domain=(0, L[1]))
T   = TensorProductSpace(comm, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
TV  = VectorSpace(T)
X  = T.local_mesh(True)
K   = T.local_wavenumbers(True,True)

# find the actual time, same as t_snap
t = tplot[ii]

# init figures
fig_q, axesq = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_q.suptitle('Potential Vorticity (q) at t = %3.0f' % t)

fig_j, axesj = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_j.suptitle('Scaled Current (Mj) at t = %3.0f' % t)

fig_p, axesp = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_p.suptitle(r'Streamfunction ($\psi$) at t = %3.0f' % t)

fig_A, axesA = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_A.suptitle('Scaled Magnetic Streamfunction (MA) at t = %3.0f' % t)

fig_L, axesL = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_L.suptitle(r'Lorentz Force ($M^2 b\cdot\nabla j$) at t = %3.0f' % t)

# init array for lorentz force component
gradj = Array(TV)

# one loop for each value of M2, updating all fields at once in figures..
for index in range(len(mag_values)):

    # pick out M2 value
    M2 = M2_vector[index]

    # set panel index, same for all figures
    axii = panelindex(index,ncols,nrows)

    # extract fields from file
    f = h5py.File(fieldfilename_array[index], "r")
    q = np.array(f['q/2D/' + str(ii)][()])  
    j = np.array(f['j/2D/' + str(ii)][()])
    A = np.array(f['A/2D/' + str(ii)][()])
    p = np.array(f['psi/2D/' + str(ii)][()])
    #u = np.array(f['u/2D/' + str(ii)][()])
    #b = np.array(f['b/2D/' + str(ii)][()])

    # add background field if necessary
    if full_field:
        q += np.array(f['q_bar/2D/' + str(0)][()])
        A += np.array(f['A_bar/2D/' + str(0)][()])
        p += np.array(f['psi_bar/2D/' + str(0)][()])
        #u += np.array(f['u_bar/2D/' + str(0)][()])
        #b += np.array(f['b_bar/2D/' + str(0)][()])

    # total mag field for lorentz force plot, compute grad(j)
    b_total = np.array(f['b/2D/' + str(ii)][()]) + np.array(f['b_bar/2D/' + str(0)][()])
    for i in range(2):
        gradj[i][:] = T.backward(1j*K[i]*T.forward(j))

    # some plotting preferences, axis labels and titles
    for axes in [axesq,axesj,axesp,axesA,axesL]:
        #axes[axii].set_aspect('equal')
        axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
        if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("x");
        if ncols==1 or (nrows==1 and axii[0]==0) or (ncols>1 and nrows>1 and axii[1]==0): axes[axii].set_ylabel("y");
        if M2 == 0: M2 = 1; # just to plot something, and not an empty panel in the hydro case

    # actually plot the fields now
    aq = axesq[axii].pcolormesh(X[0], X[1], q, cmap = 'seismic', shading = 'gouraud', \
         vmin = -np.max(abs(q)), vmax = np.max(abs(q)))
    fig_q.colorbar(aq, ax=axesq[axii])

    aj = axesj[axii].pcolormesh(X[0], X[1], np.sqrt(M2)*j, cmap = 'seismic', shading = 'gouraud', \
         vmin = -np.max(abs(np.sqrt(M2)*j)), vmax = np.max(abs(np.sqrt(M2)*j)))
    fig_j.colorbar(aj, ax=axesj[axii])

    ap = axesp[axii].pcolormesh(X[0], X[1], p, cmap = 'seismic', shading = 'gouraud', \
         vmin = -np.max(abs(p)), vmax = np.max(abs(p)))
    fig_p.colorbar(ap, ax=axesp[axii])

    aA = axesA[axii].pcolormesh(X[0], X[1], np.sqrt(M2)*A, cmap = 'seismic', shading = 'gouraud', \
         vmin = -np.max(abs(np.sqrt(M2)*A)), vmax = np.max(abs(np.sqrt(M2)*A)))
    fig_A.colorbar(aA, ax=axesA[axii])

    aL = axesL[axii].pcolormesh(X[0], X[1], M2*inner(b_total,gradj), cmap = 'seismic', shading = 'gouraud', \
         vmin = -np.max(abs(M2*inner(b_total,gradj))), vmax = np.max(abs(M2*inner(b_total,gradj))))
    fig_L.colorbar(aL, ax=axesL[axii])

# fill figures and save
fig_q.tight_layout()    
fig_q.savefig("qgmhd_q_N{}.png".format(N[0]))
fig_j.tight_layout() 
fig_j.savefig("qgmhd_j_N{}.png".format(N[0]))
fig_p.tight_layout() 
fig_p.savefig("qgmhd_p_N{}.png".format(N[0]))
fig_A.tight_layout() 
fig_A.savefig("qgmhd_A_N{}.png".format(N[0]))
fig_L.tight_layout()
fig_L.savefig("qgmhd_L_N{}.png".format(N[0]))


