#!/usr/bin/env python

# post-processing script to compute the spectral energy slopes from 
# multiple runs of the qgmhd_shenfun.py driver script to plot
# time averaged and azimuthally averaged spectra as a function
# of k (wavenumber) and in multiple cases of M2.
#
# Current outputs: 
#
# In the same directory as this script, create folders named as
# <case_prefix>-M2-<mag_value> Eg: turbulence-M2-1em4, which 
# each contain a 'output-qgmhd' output folder from the model.
# Specify the case_prefix below which is fixed for a set of 
# figure. Further, list the values of magnetism to include,
# so the suffixes to your directory names, in mag_values.
# The code will extract the actual value of M from each output.
#
# This scripts further requires a symbolic link of the code library.
# from this directory, run $ ln -s /path/to/library

# imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from shenfun import *
from scipy import ndimage
from library.data_output import fmt, panelindex

# FFTW for spectral derivatives and azimuthal averaging
try:
    import pyfftw
    from numpy import zeros as nzeros
    # Keep fft objects in cache for efficiency
    nthreads = 1
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1e8)
    def empty(N, dtype="float", bytes=16):
        return pyfftw.byte_align_empty(N, bytes, dtype=dtype)
    def zeros(N, dtype="float", bytes=16):
        return pyfftw.byte_align(nzeros(N, dtype=dtype), bytes)
    # Monkey patches for fft
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    fftn = pyfftw.interfaces.numpy_fft.fftn
except:
    print(" ")
    print(Warning("Install pyfftw, it is much faster than numpy fft"))
    print(" ")
    from numpy.fft import fftn, ifftn

# creating figures for multiple values of M^2.

# set resolution to pick out correct files
N = [1024]*2

# sets the wavenumber bounds for slope computations
k_bound = [5,10] 

# turns on/off y-axis limits on spectral slope plots
lims_on = True 
# sets values of those limits
lims = [1e-1, 1e11] 

# to compute slope from full field or pertubation
full_field = False 

# time bounds for temporal averaging
[t_begin,t_end] = [240,260] 

# set the directories you want to read from
case_prefix = "turbB0-F2-0"

# list of directory suffixes corresponding to M2 values
#mag_values = ["hydro","1em6","1em5","1em4","1em3","1em2","1em1","1em0"]
mag_values = ["hydro","1em6","1em4","1em2"]

# fixed fig size..
figsize=(5,5)

# fig format (eps,png,both)
save_format = "both"

diagfilename_array = []; fieldfilename_array = []

# create array of directory files to open
for magval in mag_values:
    diagfilename_array.append(case_prefix+"-M2-"+magval+\
        "/output-qgmhd/qgmhd_Nx{}_diagnostics.h5".format(N[0]))
    fieldfilename_array.append(case_prefix+"-M2-"+magval+\
        "/output-qgmhd/qgmhd_Nx{}_fields.h5".format(N[0]))

# initialize and show all files
print('Read from files \n', fieldfilename_array)
M2_vector = [] # init array of M2 values

# first file for invariant quantities
file0 = h5py.File(diagfilename_array[0], "r")
times  = file0['times']
twrite = file0['twrite']
t0,tf,dt = times[0], times[1], times[2]
tt       = np.arange(t0, tf, dt)
ntplot   = int(twrite[...]/times[2])
tplot    = tt[0::ntplot]
assert(tf>=t_end),"Avg Time Range Too Large: Outside Simulation Time Range"
domain = file0['domain']
N      = file0['size']
F2     = file0['F2'][...]
Uj     = file0['Uj'][...]
assert(full_field*Uj == 0), "Avoid computing the full spectrum for the jet, Streamfunction is not periodic."

# temporal averaging limits
first = int(np.where(np.array(tplot)==t_begin)[0])
last = int(np.where(np.array(tplot)==t_end)[0])

print('Opening diag files to pull M2 values...')

for filename in diagfilename_array:
    filed = h5py.File(filename, "r")
    M2_vector.append(float(filed['M2'][...]))

print(M2_vector)

# first file for invariant domain quantities
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

# maximum wavenumbers
nxh = int(N[0]/2)
nyh = int(N[1]/2)    

kx = np.fft.fftfreq(N[0], d=L[0]/(2.*np.pi))*N[0]
ky = np.fft.fftfreq(N[1], d=L[1]/(2.*np.pi))*N[1]
kxp = np.fft.fftshift(kx)
kyp = np.fft.fftshift(ky)

# Compute a radial coordinate

[kxxp,kyyp]= np.meshgrid(kxp,kyp)
R          = np.sqrt(kxxp**2 + kyyp**2)

# Azimuthal avg
dbin = 1.0
kbin = 2.*np.pi/L[0]*np.arange(0., nxh+1, dbin)

np.seterr(divide='ignore', invalid='ignore')

##################
# To rescale quantity by expected slope
KEmult = 1+0*kbin[1:]**(5/3)
MEmult = 1+0*KEmult
PEmult = 1+0*KEmult
TEmult = 1+0*kbin[1:]**(3/2)
##################

KE_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
ME_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
PE_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
TE_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])

## Main loop to create averaged spectra..

for index in range(len(mag_values)):

    print("Now computing M2 = %1e" % M2_vector[index])
    M2 = M2_vector[index]

    for ii in range(first,last+1):
    
        f = h5py.File(fieldfilename_array[index], "r")
        t = tplot[ii]
        print("time: "+str(t))

        psi = np.array(f['psi/2D/' + str(ii)][()])
        u = np.array(f['u/2D/' + str(ii)][()])
        b = np.array(f['b/2D/' + str(ii)][()])

        if full_field:
            psi += np.array(f['psi_bar/2D/' + str(0)][()])
            u += np.array(f['u_bar/2D/' + str(0)][()])
            b += np.array(f['b_bar/2D/' + str(0)][()])

        u_hat = fftn(u[0])
        v_hat = fftn(u[1])
        psi_hat = fftn(psi)
        b1_hat = fftn(b[0])
        b2_hat = fftn(b[1])

        # to actually be the spectrum we talk about in spectral fluxes
        scale = 0.5*L[0]*L[1]/(4*np.pi**2)

        PEspectrum = scale*np.fft.fftshift(abs(psi_hat)**2)
        KEspectrum = scale*np.fft.fftshift(abs(u_hat)**2+abs(v_hat)**2)
        MEspectrum = scale*np.fft.fftshift(abs(b1_hat)**2+abs(b2_hat)**2)
        TEspectrum = KEspectrum + F2*PEspectrum +  M2*MEspectrum

        KEbin = np.zeros(kbin.shape)
        MEbin = np.zeros(kbin.shape)
        PEbin = np.zeros(kbin.shape)
        TEbin = np.zeros(kbin.shape)

        for jj in range(0, len(kbin)-1):
            mask = np.logical_and(R >= kbin[jj], R < kbin[jj+1]).astype(int)
            theta_int = 2*np.pi*kbin[jj]
            PEbin[jj] = ndimage.mean(PEspectrum, mask)*theta_int
            KEbin[jj] = ndimage.mean(KEspectrum, mask)*theta_int
            MEbin[jj] = ndimage.mean(MEspectrum, mask)*theta_int
            TEbin[jj] = ndimage.mean(TEspectrum, mask)*theta_int
    
        KE_vec[index,ii] = KEbin
        ME_vec[index,ii] = MEbin
        PE_vec[index,ii] = PEbin
        TE_vec[index,ii] = TEbin

## create averaged figure..
fig_TE, axesTE = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)
#fig_TE.suptitle(r'Total Energy Spectrum')

fig_KE, axesKE = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)
#fig_KE.suptitle(r'Kinetic Energy Spectrum')

fig_PE, axesPE = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)
#fig_PE.suptitle(r'Potential Energy Spectrum')

fig_ME, axesME = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)
#fig_ME.suptitle(r'Magnetic Energy Spectrum')

print("Finished Creating Datasets, now averaging...")

### one loop iteration for each value of M and in each plot all fields..

color_index = ['-k','-r','-g','-b']
lw = 2

for index in range(len(mag_values)):

    KEbin = np.mean(KE_vec[index,first:last+1,:],axis=0)
    MEbin = np.mean(ME_vec[index,first:last+1,:],axis=0)
    PEbin = np.mean(PE_vec[index,first:last+1,:],axis=0)
    TEbin = np.mean(TE_vec[index,first:last+1,:],axis=0)

    I1 = np.array( np.where(((kbin > k_bound[0]) & (kbin < k_bound[1]))) )
    I1 = I1.ravel()
    if len(I1) > 2:
        p1 = np.polyfit(np.log10(kbin[I1]), np.log10(KEbin[I1]), 1)
        p2 = np.polyfit(np.log10(kbin[I1]), np.log10(MEbin[I1]), 1)
        p3 = np.polyfit(np.log10(kbin[I1]), np.log10(TEbin[I1]), 1)
        p4 = np.polyfit(np.log10(kbin[I1]), np.log10(PEbin[I1]), 1)

    axesKE.loglog(kbin[1:], KEmult*KEbin[1:], color_index[index], linewidth = lw, \
        label=r'$M^2 = $'+fmt(M2_vector[index])+r', $\alpha = $ % 2.2f' % p1[0])
    #axesKE.loglog(kbin[1:], KEmult*pow(10,np.polyval(p1,np.log10(kbin[1:]))), '-k')
    #axesKE.loglog(kbin[I1[0]], TEmult[I1[0]]*pow(10,np.polyval(p1,np.log10(kbin[I1[0]]))), 'og')
    #axesKE.loglog(kbin[I1[-1]], TEmult[I1[-1]]*pow(10,np.polyval(p1,np.log10(kbin[I1[-1]]))), 'og')

    axesPE.loglog(kbin[1:], PEmult*PEbin[1:], color_index[index], linewidth = lw, \
        label=r'$M^2 = $'+fmt(M2_vector[index])+r', $\alpha = $ % 2.2f' % p4[0])
    #axesPE.loglog(kbin[1:], PEmult*pow(10,np.polyval(p4,np.log10(kbin[1:]))), '-k')
    #axesPE.loglog(kbin[I1[0]], PEmult[I1[0]]*pow(10,np.polyval(p4,np.log10(kbin[I1[0]]))), 'og')
    #axesPE.loglog(kbin[I1[-1]], PEmult[I1[-1]]*pow(10,np.polyval(p4,np.log10(kbin[I1[-1]]))), 'og')

    axesME.loglog(kbin[1:], MEmult*MEbin[1:], color_index[index], linewidth = lw, \

        label=r'$M^2 = $'+fmt(M2_vector[index])+r', $\alpha = $ % 2.2f' % p2[0])
    #axesME.loglog(kbin[1:], MEmult*pow(10,np.polyval(p2,np.log10(kbin[1:]))), '-k')
    #axesME.loglog(kbin[I1[0]], MEmult[I1[0]]*pow(10,np.polyval(p2,np.log10(kbin[I1[0]]))), 'og')
    #axesME.loglog(kbin[I1[-1]], MEmult[I1[-1]]*pow(10,np.polyval(p2,np.log10(kbin[I1[-1]]))), 'og')

    axesTE.loglog(kbin[1:], TEmult*TEbin[1:], color_index[index], linewidth = lw, \
        label=r'$M^2 = $'+fmt(M2_vector[index])+r', $\alpha = $ % 2.2f' % p3[0])
    #axesTE.loglog(kbin[1:], TEmult*pow(10,np.polyval(p3,np.log10(kbin[1:]))), '-k')
    #axesTE.loglog(kbin[I1[0]], TEmult[I1[0]]*pow(10,np.polyval(p3,np.log10(kbin[I1[0]]))), 'og')
    #axesTE.loglog(kbin[I1[-1]], TEmult[I1[-1]]*pow(10,np.polyval(p3,np.log10(kbin[I1[-1]]))), 'og')

axesTE.set_ylabel(r"$\hat E$");
axesKE.set_ylabel(r"$\hat E_K$");
axesME.set_ylabel(r"$\hat E_M$");
axesPE.set_ylabel(r"$\hat E_P$");

for axes in [axesTE,axesKE,axesME,axesPE]:
    axes.set_xlabel("wavenumber (k)");
    axes.grid(True)
    if lims_on: axes.set_ylim(lims);
    axes.legend(loc='lower left')

for fig in [fig_TE, fig_PE, fig_KE, fig_ME]:
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.3)
    #fig.legend(loc='lower center',fancybox=True, shadow=True, ncol=2,fontsize=9)

if save_format == 'png' or save_format == 'both':
    fig_TE.savefig("qgmhd_TE_N{}.png".format(N[0]))
    fig_PE.savefig("qgmhd_PE_N{}.png".format(N[0]))
    fig_KE.savefig("qgmhd_KE_N{}.png".format(N[0]))
    fig_ME.savefig("qgmhd_ME_N{}.png".format(N[0]))
if save_format == 'eps' or save_format == 'both':
    fig_TE.savefig("qgmhd_TE_N{}.eps".format(N[0]))
    fig_PE.savefig("qgmhd_PE_N{}.eps".format(N[0]))
    fig_KE.savefig("qgmhd_KE_N{}.eps".format(N[0]))
    fig_ME.savefig("qgmhd_ME_N{}.eps".format(N[0]))

print("Done.")


