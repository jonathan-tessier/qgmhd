#!/usr/bin/env python

# post-processing script to read the fields output from 
# multiple runs of the qgmhd_shenfun.py driver script to plot
# snapshots of them as a function of F (inv. Burger num) and time.
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
from library.data_output import fmt

# creating figures for multiple values of M^2 and t.

# set resolution to pick out correct files
N = [1024]*2

# Flag to plot the full fields (q = \bar q + q') or perturbations (q')
full_field = False

# set the directories you want to read from
case_prefix = "turbB0"

# set the magnetic case to plot
m_value = "hydro"

# turns the colorbars on/off
cb = False

# selects the times to plot
time_values = [20,50,100,150]

# selects the F cases to plot
F_values = ["0","025","05","1"]

# figure panel config
ncols,  nrows  = [int(len(time_values)), int(len(F_values))];
hscale, vscale = [2.0,2.0];
figsize=((hscale+cb*0.5)*ncols, (vscale+0.05)*nrows)

diagfilename_array = []
fieldfilename_array = []

# create array of directory files to open
for Fval in F_values:
    diagfilename_array.append(case_prefix+"-F2-"+Fval+"-M2-"+m_value+"/output-qgmhd/qgmhd_Nx{}_diagnostics.h5".format(N[0]))
    fieldfilename_array.append(case_prefix+"-F2-"+Fval+"-M2-"+m_value+"/output-qgmhd/qgmhd_Nx{}_fields.h5".format(N[0]))

print('Read from files \n', fieldfilename_array)

F2_vector = [] # init dirs

# first file for invariant quantities
file0 = h5py.File(diagfilename_array[0], "r")
times  = file0['times']
twrite = file0['twrite']
t0,tf,dt = times[0], times[1], times[2]
tt       = np.arange(t0, tf, dt)
ntplot   = int(twrite[...]/times[2])
tplot    = tt[0::ntplot]
assert(tf>=np.max(time_values)),"Snapshot Time Too Large: Outside Simulation Time Range"

# to store indices of desired times
index_vector = []
for t_snap in time_values:
    index_vector.append(int(np.where(np.array(tplot)==t_snap)[0]))
print(index_vector)

domain = file0['domain']
M2     = file0['M2'][...]

print('Opening diag files to pull M2 values...')

for filename in diagfilename_array:
    filed = h5py.File(filename, "r")
    F2_vector.append(float(filed['F2'][...]))

print(F2_vector)

# first file for invariant domain quantities
f = h5py.File(fieldfilename_array[0], "r")
keys = list(f.keys()) # list variables

# Which variable to view
var1 = keys[0]

x0 = np.array(f[var1 + '/domain/x0'][()])       # bounds in 0-axis
x1 = np.array(f[var1 + '/domain/x1'][()])       # bounds in 1-axis

L = (x0[-1], x1[-1])

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

fig_q, axesq = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_q.suptitle('Potential Vorticity (q)')

fig_j, axesj = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_j.suptitle('Scaled Current (Mj)')

fig_p, axesp = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_p.suptitle(r'Streamfunction ($\psi$)')

fig_A, axesA = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_A.suptitle('Scaled Magnetic Streamfunction (MA)')

fig_L, axesL = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig_L.suptitle(r'Lorentz Force ($M^2 b\cdot\nabla j$)')

### one loop for each field to plot..

for findex in range(len(F_values)):
    # coefficient of the field being plotted
    if M2 == 0:
        M2 = 1 # in the hydro case, this will plot passive field
    print("Now computing F2 = "+str(F2_vector[findex]))

    for tindex in range(len(time_values)):

        ii = index_vector[tindex]
        t = tplot[ii]

        f = h5py.File(fieldfilename_array[findex], "r")
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

        gradj = Array(TV)
        b_total = np.array(f['b/2D/' + str(ii)][()]) + np.array(f['b_bar/2D/' + str(0)][()])
        for i in range(2):
            gradj[i][:] = T.backward(1j*K[i]*T.forward(j))

        for axes in [axesq,axesj,axesp,axesA,axesL]:
            if findex == 0: axes[findex,tindex].set_title("t = {}".format(time_values[tindex]));
            if tindex == 0: axes[findex,tindex].set_ylabel("F = "+str(F2_vector[findex]));
            for fi in range(len(F_values)):
                for ti in range(len(time_values)):
                    #if not(cb): axes[fi,ti].set_aspect('equal');
                    axes[fi,ti].set_xticklabels([])
                    axes[fi,ti].set_yticklabels([])

        vmin = -np.max(abs(q));
        vmax = np.max(abs(q));
        aq = axesq[findex,tindex].pcolormesh(X[0], X[1], q, cmap = 'seismic', shading = 'gouraud', \
             vmin = vmin, vmax = vmax)
        if cb: fig_q.colorbar(aq, ax=axesq[findex,tindex]);

        vmin = -np.max(abs(np.sqrt(M2)*j));
        vmax = np.max(abs(np.sqrt(M2)*j));
        aj = axesj[findex,tindex].pcolormesh(X[0], X[1], np.sqrt(M2)*j, cmap = 'seismic', shading = 'gouraud', \
             vmin = vmin, vmax = vmax)
        if cb: fig_j.colorbar(aj, ax=axesj[findex,tindex]);

        vmin = -np.max(abs(p));
        vmax = np.max(abs(p));
        ap = axesp[findex,tindex].pcolormesh(X[0], X[1], p, cmap = 'seismic', shading = 'gouraud', \
             vmin = vmin, vmax = vmax)
        if cb: fig_p.colorbar(ap, ax=axesp[findex,tindex]);

        vmin = -np.max(abs(np.sqrt(M2)*A));
        vmax = np.max(abs(np.sqrt(M2)*A));
        aA = axesA[findex,tindex].pcolormesh(X[0], X[1], np.sqrt(M2)*A, cmap = 'seismic', shading = 'gouraud', \
             vmin = vmin, vmax = vmax)
        if cb: fig_A.colorbar(aA, ax=axesA[findex,tindex]);

        vmin = -np.max(abs(M2*inner(b_total,gradj)));
        vmax = np.max(abs(M2*inner(b_total,gradj)));
        aL = axesL[findex,tindex].pcolormesh(X[0], X[1], M2*inner(b_total,gradj), cmap = 'seismic', shading = 'gouraud', \
             vmin = vmin, vmax = vmax)
        if cb: fig_L.colorbar(aL, ax=axesL[findex,tindex]);

for fig in [fig_q,fig_j,fig_p,fig_A,fig_L]: 
    fig.tight_layout()
    if  not(cb): fig.subplots_adjust(hspace = 0.0, wspace=0.0);

fig_q.savefig("qgmhd_q_F_snaps_N{}.png".format(N[0]))
fig_j.savefig("qgmhd_j_F_snaps_N{}.png".format(N[0]))
fig_p.savefig("qgmhd_p_F_snaps_N{}.png".format(N[0]))
fig_A.savefig("qgmhd_A_F_snaps_N{}.png".format(N[0]))
fig_L.savefig("qgmhd_L_F_snaps_N{}.png".format(N[0]))


