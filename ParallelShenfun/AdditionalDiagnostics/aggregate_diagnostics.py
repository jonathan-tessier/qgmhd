#!/usr/bin/env python

# post-processing script to read the diagnostics output from 
# multiple runs of the qgmhd_shenfun.py driver script to plot
# diagnostics as a function of M and time (mag. field strength)
#
# Current outputs: energy, anisotropy, lorentz force, microscales
# growth rates, conserved quantities 
#
# In the same directory as this script, create folders named as
# <case_prefix>-M2-<mag_value> Eg: turbulence-M2-1em4, which 
# each contain a 'output-qgmhd' output folder from the model.
# Specify the case prefix below which is fixed for a set of 
# figure. Further, list the values of magnetism to include,
# so the suffixes to your directory names, in mag_values.
# The code will extract the actual value of M from each output.

#imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from library.data_output import fmt, panelindex

# set resolution to pick out correct files
N = [1024]*2

# set the directories you want to read from
case_prefix = "turbB0-F2-0"

# temporal range for growth rate computation
t_bound = [30,38]

# list of directory suffixes corresponding to M2 values 
#mag_values = ["hydro","1em6","1em5","1em4","1em3","1em2","1em1","1em0"]
mag_values = ["hydro","1em6","1em4","1em2"]

# figure panel config
ncols,  nrows  = [int(len(mag_values)),1] # rows and cols
hscale, vscale = [2.5,2.5]; # approximate size of each panel
figsize=(hscale*ncols,vscale*nrows + 1) # actual figure size

# check the figure will actually fit all values
assert(ncols*nrows == int(len(mag_values))),"Number of simulations doesn't fit in the grid."
filename_array = [] # init dir

# create array of directory files to open
for magval in mag_values:
    filename_array.append(case_prefix+"-M2-"+magval+\
        "/output-qgmhd/qgmhd_Nx{}_realdiagnostics.h5".format(N[0]))

# initialize and show all files
print('Read from files \n', filename_array)
file_array = []; M2_vector = [] # init dirs
KE_vector    = []; 
PE_vector    = []; 
Q2_vector    = []; 
Q1_vector    = []; 
qgrow_vector = []; 
ME_vector    = []; 
A2_vector    = []; 
A1_vector    = []; 
j2_vector    = []; 
Agrow_vector = []; 
Hel_vector   = []; 
TE_vector    = []; 
Lu_vector    = []; 
Lb_vector    = []; 
u12_vector   = [];
u22_vector   = [];
b12_vector   = [];
b22_vector   = [];
qx2_vector   = [];
qy2_vector   = [];
jx2_vector   = [];
jy2_vector   = [];
lor_vector   = [];

# first file for invariant quantities
file0 = h5py.File(filename_array[0], "r")
times  = file0['times']
tt     = np.arange(times[0], times[1], times[2])
domain = file0['domain']
N      = file0['size']
F2     = file0['F2']
Uj     = file0['Uj']

print('Opening files...')

# for 0/0 current diagnostics
np.seterr(divide='ignore', invalid='ignore')

# looping through files, pick out the M value and store the diagnostics
for filename in filename_array:
    file1 = h5py.File(filename, "r")
    file_array.append(file1)
    M2_vector.append(float(file1['M2'][...]))
    vals = file1['values']

    KE_vector.append(vals[:,0])
    PE_vector.append(vals[:,1])
    Q2_vector.append(vals[:,2])
    Q1_vector.append(vals[:,3])
    qgrow_vector.append(vals[:,4])
    ME_vector.append(vals[:,5])
    A2_vector.append(vals[:,6])
    A1_vector.append(vals[:,7])
    j2_vector.append(vals[:,8])
    Agrow_vector.append(vals[:,9])
    Hel_vector.append(vals[:,10])
    TE_vector.append(vals[:,0]+vals[:,1]+vals[:,5])
    Lu_vector.append( np.sqrt(vals[:,11]+vals[:,12])/np.sqrt(vals[:,2]))
    Lb_vector.append( np.sqrt(vals[:,13]+vals[:,14])/np.sqrt(vals[:,8]))
    u12_vector.append(vals[:,11])
    u22_vector.append(vals[:,12])
    b12_vector.append(vals[:,13])
    b22_vector.append(vals[:,14])
    qx2_vector.append(vals[:,15])
    qy2_vector.append(vals[:,16])
    jx2_vector.append(vals[:,17])
    jy2_vector.append(vals[:,18])
    lor_vector.append(vals[:,19])

# show M values
print(M2_vector)

# Now for the plotting

### Energy

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=True, figsize=figsize)
fig.suptitle('Energy Decomposition')

for index in range(len(mag_values)):
    axii = panelindex(index,ncols,nrows) # axes index
    axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
    axes[axii].plot(tt, TE_vector[index][:-1], lw=3, color='k', label='Total')
    axes[axii].plot(tt, KE_vector[index][:-1], lw=3, color='r', label='Kinetic', dashes=[6,2])
    axes[axii].plot(tt, PE_vector[index][:-1], lw=3, color='g', label='Potential', dashes=[6,2])
    axes[axii].plot(tt, ME_vector[index][:-1], lw=3, color='b', label='Magnetic', dashes=[6,2])
    axes[axii].grid(True)
    if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("t");

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',fancybox=True, shadow=True, ncol=4)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

print('Generated Energy Figure.')

file_name_png = "qgmhd_energy_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

### Anisotropy measure

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig.suptitle('Anisotropy Norm')

for index in range(len(mag_values)):
    axii = panelindex(index,ncols,nrows) # axes index
    axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
    axes[axii].plot(tt, u12_vector[index][:-1]/(u12_vector[index][:-1]+u22_vector[index][:-1]), \
        lw=3, color='r', label=r'$\langle u_1^2\rangle /\langle u_1^2+u_2^2\rangle$')
    axes[axii].plot(tt, b12_vector[index][:-1]/(b12_vector[index][:-1]+b22_vector[index][:-1]), \
        lw=3, color='b', label=r'$\langle b_1^2\rangle/\langle b_1^2+b_2^2\rangle$')
    axes[axii].plot(tt, qx2_vector[index][:-1]/(qx2_vector[index][:-1]+qy2_vector[index][:-1]), \
        lw=3, color='c', label=r'$\langle q_x^2\rangle/\langle q_x^2+q_y^2\rangle$')
    axes[axii].plot(tt, jx2_vector[index][:-1]/(jx2_vector[index][:-1]+jy2_vector[index][:-1]), \
        lw=3, color='m', label=r'$\langle j_x^2\rangle/\langle j_x^2+j_y^2\rangle$')
    axes[axii].grid(True)
    if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("t");
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',fancybox=True, shadow=True, ncol=4)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
print('Generated Anisotropy Figure.')

file_name_png = "qgmhd_anisotropy_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

### Lorentz Force measure

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=True, figsize=figsize)
fig.suptitle('Lorentz Force Norm')

for index in range(len(mag_values)):
    axii = panelindex(index,ncols,nrows) # axes index
    axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
    axes[axii].plot(tt, lor_vector[index][:-1], lw=3, color='b', \
        label=r'$M^2 \sqrt{\langle(b\cdot\nabla j)^2\rangle}$')
    axes[axii].grid(True)
    if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("t");
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',fancybox=True, shadow=True, ncol=4)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
print('Generated Lorentz Figure.')

file_name_png = "qgmhd_lorentz_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

### Conserved quantitites

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, sharex=True, figsize=figsize)
fig.suptitle('Relative Error in Conserved Quantities')

for index in range(len(mag_values)):
    axii = panelindex(index,ncols,nrows) # axes index
    axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
    twin1 = axes[axii].twinx()
    Hplot = (Hel_vector[index][:-1]-Hel_vector[index][0])/Hel_vector[index][0]
    Aplot = (A2_vector[index][:-1]-A2_vector[index][0])/A2_vector[index][0]
    plt1, = axes[axii].plot(tt, Hplot, lw=3, color='navy', label=r'$(H-H(0))/H(0)$')
    plt2, = twin1.plot(tt, Aplot, lw=3, color='m', label=r'$(A^2-A^2(0))/A^2(0)$')
    axes[axii].grid(True)
    if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("t");
    axes[axii].tick_params(axis='y', colors=plt1.get_color())
    twin1.tick_params(axis='y', colors=plt2.get_color())
fig.legend(handles=[plt1,plt2], loc='lower center',fancybox=True, shadow=True, ncol=4)
fig.tight_layout()
fig.subplots_adjust(wspace=0.55,bottom=0.25)
print('Generated Conservation Figure.')

file_name_png = "qgmhd_conserv_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

### Growth rate

I1 = np.array( np.where(((tt > t_bound[0]) & (tt < t_bound[1]))) )
I1 = I1.ravel()
p1_array = []; pp1_array = []; p2_array = []; pp2_array = [];

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig.suptitle('Perturbation Norms')

if len(I1) > 2:
    for index in range(len(mag_values)):
        axii = panelindex(index,ncols,nrows)
        p1 = np.polyfit(tt[I1], np.log(qgrow_vector[index][I1]), 1)
        pp1 = np.exp(np.polyval(p1,(tt)))
        p2 = np.polyfit(tt[I1], np.log(Agrow_vector[index][I1]), 1)
        pp2 = np.exp(np.polyval(p2,(tt)))
        print('q and A growth rates for '+str(M2_vector[index])+' are ', p1[0], p2[0])
        axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
        plt1, = axes[axii].semilogy(tt, qgrow_vector[index][:-1],  lw=3, color='r', label=r'$||q-\bar{q}||$')
        plt2, = axes[axii].semilogy(tt, Agrow_vector[index][:-1],  lw=3, color='b', label=r'$||A-\bar{A}||$')
        plt3, = axes[axii].semilogy(tt[I1], pp1[I1], '--m',  lw=3, label="q slope: {p:8.4f}".format(p=p1[0]))
        plt4, = axes[axii].semilogy(tt[I1], pp2[I1], '--c',  lw=3, label="A slope: {p:8.4f}".format(p=p2[0]))
        if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("t");        
        axes[axii].legend(handles = [plt3,plt4], loc='lower right')
        axes[axii].grid(True)
else:
    raise(IndexError("Growth rate interpolation range too small."))
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles[0:2], labels[0:2], loc='lower center',fancybox=True, shadow=True, ncol=2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
print('Generated Growth Rates Figure.')

file_name_png = "qgmhd_growthrates_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

### L_u and L_b

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=figsize)
fig.suptitle('Microscales')

for index in range(len(mag_values)):
    axii = panelindex(index,ncols,nrows)
    axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))
    axes[axii].semilogy(tt, Lu_vector[index][:-1], lw=3, color='r', \
        label=r'$L_u = \langle {\bf u\cdot u}\rangle^{1/2}/\langle q^2\rangle^{1/2}$')
    axes[axii].semilogy(tt, Lb_vector[index][:-1], lw=3, color='b', \
        label=r'$L_b = \langle {\bf b\cdot b}\rangle^{1/2}/\langle j^2\rangle^{1/2}$')
    axes[axii].grid(True)
    if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("t");
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',fancybox=True, shadow=True, ncol=4)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
print('Generated Microscales Figure.')

file_name_png = "qgmhd_microscales_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)


