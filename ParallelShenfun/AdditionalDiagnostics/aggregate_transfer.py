#!/usr/bin/env python

# post-processing script to compute the spectral energy transfer from 
# multiple runs of the qgmhd_shenfun.py driver script to plot
# time averaged and azimuthal averaged version of them as a function
# of k (wavenumbers) and in multiple cases of M2.
#
# Current outputs: T(k) components, viscous and full equations
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
from scipy import ndimage, integrate
from library.operators import inner
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

# time bounds for temporal averaging
[t_begin,t_end] = [100,300] 

# set the directories you want to read from
case_prefix = "turbB0-F2-0"

# list of directory suffixes corresponding to M2 values 
#mag_values = ["hydro","1em6","1em5","1em4","1em3","1em2","1em1","1em0"]
mag_values = ["hydro","1em6","1em4","1em2"]

# figure panel config
ncols,  nrows  = [int(len(mag_values)), 1];
hscale, vscale = [2.5,2.5];
figsize=((hscale+0.3)*ncols, vscale*nrows+1.2)

# check the figure will actually fit all values
assert(ncols*nrows == int(len(mag_values))),"Number of simulations doesn't fit in the grid."

#init dir arrays
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
B0     = file0['B0'][...]
Re     = file0['Re'][...]
Rm     = file0['Rm'][...]

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
T   = TensorProductSpace(MPI.COMM_SELF, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
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
[kxx, kyy] = np.meshgrid(kx,ky)
[kxxp,kyyp]= np.meshgrid(kxp,kyp)
K          = np.sqrt(kxx**2 + kyy**2)
KP         = np.sqrt(kxxp**2 + kyyp**2)

np.seterr(divide='ignore', invalid='ignore')

iK     = np.zeros((2,N[1],N[0]), dtype=complex)
iK[0]  = 1j*kxx
iK[1]  = 1j*kyy

if F2 == 0:
    print("F is zero")
    K2oK2F2  = 1
else:
    K2oK2F2  = np.where((K**2 + F2) == 0.0, 0.0, K**2/(K**2+F2)).astype(float)
    print("F is non-zero")

# Azimuthal avg
dbin = 1.0
kbin = 2.*np.pi/L[0]*np.arange(0., nxh+1, dbin)

HEadv_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
HElor_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
HEvis_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEadv_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEvis_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
HEtot_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEtot_vec = np.zeros([len(mag_values),len(tplot),len(kbin)])

#constants for flux computations
Lo2p2 = (L[0]*L[1]/(4*np.pi**2))

## Main loop to create averaged spectra..
for index in range(len(mag_values)):

    print("Now computing M2 = %1e" % M2_vector[index])

    for ii in range(first,last+1):
    
        f = h5py.File(fieldfilename_array[index], "r")
        t = tplot[ii]
        print("time: "+str(t))

        q = np.array(f['q/2D/' + str(ii)][()])  
        j = np.array(f['j/2D/' + str(ii)][()])
        A = np.array(f['A/2D/' + str(ii)][()])
        upert = np.array(f['u/2D/' + str(ii)][()])
        bpert = np.array(f['b/2D/' + str(ii)][()])
        p = np.array(f['psi/2D/' + str(ii)][()])

        u = upert + np.array(f['u_bar/2D/' + str(0)][()])
        b = bpert + np.array(f['b_bar/2D/' + str(0)][()])
       
        q_hat = fftn(q)
        j_hat = fftn(j)
        A_hat = fftn(A)
        psi_hat = fftn(p)

        gradq = zeros((2,N[1],N[0]))
        gradj = zeros((2,N[1],N[0]))
        gradA = zeros((2,N[1],N[0]))

        ### Compute
   
        for i in range(2):
            gradq[i] = (ifftn( iK[i]*q_hat)).real
            gradj[i] = (ifftn( iK[i]*j_hat)).real  
            gradA[i] = (ifftn( iK[i]*A_hat)).real    

        HE_advect  = +Lo2p2*(psi_hat*np.conjugate(fftn(inner(u,gradq)))).real
        HE_lorentz = -M2_vector[index]*Lo2p2*(psi_hat*np.conjugate(fftn(inner(b,gradj)))).real
        HE_viscous = -2*K2oK2F2/Lo2p2*(q_hat*np.conjugate(q_hat)/Re).real
        ME_advect  = +M2_vector[index]*Lo2p2*(j_hat*np.conjugate(fftn(inner(u,gradA)-B0*upert[1]))).real
        ME_viscous = -M2_vector[index]*2/Lo2p2*(j_hat*np.conjugate(j_hat)/Rm).real
        HE_total   = HE_advect + HE_lorentz + HE_viscous
        ME_total   = ME_advect + ME_viscous
        
        HE_advect_bin  =  np.zeros(kbin.shape)
        HE_lorentz_bin =  np.zeros(kbin.shape)
        HE_viscous_bin =  np.zeros(kbin.shape) 
        ME_advect_bin  =  np.zeros(kbin.shape)
        ME_viscous_bin =  np.zeros(kbin.shape)
        HE_total_bin   =  np.zeros(kbin.shape)
        ME_total_bin   =  np.zeros(kbin.shape)

        for jj in range(0, len(kbin)-1):
            mask = np.logical_and(KP >= kbin[jj], KP < kbin[jj+1]).astype(int)
            theta_int = 2*np.pi*kbin[jj]
            HE_advect_bin[jj] = ndimage.mean(np.fft.fftshift(HE_advect).real, mask)*theta_int
            HE_lorentz_bin[jj] = ndimage.mean(np.fft.fftshift(HE_lorentz).real, mask)*theta_int
            HE_viscous_bin[jj] = ndimage.mean(np.fft.fftshift(HE_viscous).real, mask)*theta_int
            ME_advect_bin[jj] = ndimage.mean(np.fft.fftshift(ME_advect).real, mask)*theta_int
            ME_viscous_bin[jj] = ndimage.mean(np.fft.fftshift(ME_viscous).real, mask)*theta_int
            HE_total_bin[jj] = ndimage.mean(np.fft.fftshift(HE_total).real, mask)*theta_int
            ME_total_bin[jj] = ndimage.mean(np.fft.fftshift(ME_total).real, mask)*theta_int

        # store for temporal averaging

        HEadv_vec[index,ii,:] = HE_advect_bin 
        HElor_vec[index,ii,:] = HE_lorentz_bin
        MEadv_vec[index,ii,:] = ME_advect_bin
        HEvis_vec[index,ii,:] = HE_viscous_bin
        MEvis_vec[index,ii,:] = ME_viscous_bin
        HEtot_vec[index,ii,:] = HE_total_bin
        MEtot_vec[index,ii,:] = ME_total_bin

## create averaged figure..

fig_Dyn, axesDyn = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=figsize)
fig_Dyn.suptitle('Dynamic Transfers')

fig_Vis, axesVis = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=figsize)
fig_Vis.suptitle('Viscous Transfers')

fig_Tot, axesTot = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=figsize)
fig_Tot.suptitle('HE and ME Transfers')

print("Finished Creating Datasets, now averaging...")

# temporal averaging limits

first = int(np.where(np.array(tplot)==t_begin)[0])
last = int(np.where(np.array(tplot)==t_end)[0])

### one loop iteration for each value of M and in each plot all fields..

for index in range(len(mag_values)):

    axii = panelindex(index,ncols,nrows)

    HEvis = np.mean(HEvis_vec[index,first:last+1,:],axis=0)
    MEvis = np.mean(MEvis_vec[index,first:last+1,:],axis=0)
    HEadv = np.mean(HEadv_vec[index,first:last+1,:],axis=0)
    HElor = np.mean(HElor_vec[index,first:last+1,:],axis=0)
    MEadv = np.mean(MEadv_vec[index,first:last+1,:],axis=0)
    HEtot = np.mean(HEtot_vec[index,first:last+1,:],axis=0)
    MEtot = np.mean(MEtot_vec[index,first:last+1,:],axis=0)

    p1, = axesDyn[axii].plot(kbin, HEadv, '-r', label=r'$ T_{q}(k) $')
    p3, = axesDyn[axii].plot(kbin, HElor, '--r', label=r'$ T_{L}(k) $')

    twin1 = axesDyn[axii].twinx()
    p4, = twin1.plot(kbin, MEadv, '--b', label=r'$ T_{A}(k) $')

    twin2 = axesVis[axii].twinx() 
    p2, = axesVis[axii].plot(kbin, HEvis, '-r', label=r'$-2R_e^{-1} k^2\hat E_H$')
    p5, = twin2.plot(kbin, MEvis, '-b', label=r'$-2R_m^{-1} k^2\hat E_M$')

    twin3 = axesTot[axii].twinx() 
    p6, = axesTot[axii].plot(kbin, HEtot, '-r', label=r'$T_q + T_L - 2R_e^{-1}k^2\hat E_H$')
    p7, = twin3.plot(kbin, MEtot, '-b', label=r'$T_A - 2R_m^{-1}k^2\hat E_M $')
    
    for axes in [axesDyn,axesVis,axesTot]:
        axes[axii].set_xscale('log')
        if nrows==1 or (nrows>1 and axii[0]==nrows-1): axes[axii].set_xlabel("k");
        axes[axii].grid(True)
        axes[axii].tick_params(axis='x')
        axes[axii].set_title(r"$M^2 = $"+fmt(M2_vector[index]))

    for [axes,plots] in [[axesDyn[axii],p1],[twin1,p4],[axesVis[axii],p2],[twin2,p5],[axesTot[axii],p6],[twin3,p7]]:
        axes.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axes.tick_params(axis='y', colors=plots.get_color())

    pltmax1 = max(np.max(abs(HEadv)),np.max(abs(HElor)))
    axesDyn[axii].set_ylim(-pltmax1,pltmax1)
    twin1.set_ylim(-np.max(abs(MEadv)),np.max(abs(MEadv)))

    axesVis[axii].set_ylim(-np.max(abs(HEvis)),0)
    twin2.set_ylim(-np.max(abs(MEvis)),0)

    axesTot[axii].set_ylim(-np.max(abs(HEtot)),np.max(abs(HEtot)))
    twin3.set_ylim(-np.max(abs(MEtot)),np.max(abs(MEtot)))

fig_Dyn.legend(handles=[p1,p3,p4], loc='lower center',fancybox=True, shadow=True, ncol=3)
fig_Vis.legend(handles=[p2,p5], loc='lower center',fancybox=True, shadow=True, ncol=2)
fig_Tot.legend(handles=[p6,p7], loc='lower center',fancybox=True, shadow=True, ncol=2)

fig_Dyn.tight_layout()
fig_Dyn.subplots_adjust(wspace=0.5, bottom=0.25)
fig_Dyn.savefig("qgmhd_DynTran_N{}.png".format(N[0]))
fig_Vis.tight_layout()
fig_Vis.subplots_adjust(wspace=0.5, bottom=0.25)
fig_Vis.savefig("qgmhd_VisTran_N{}.png".format(N[0]))
fig_Tot.tight_layout()
fig_Tot.subplots_adjust(wspace=0.5, bottom=0.25)
fig_Tot.savefig("qgmhd_TotTran_N{}.png".format(N[0]))
print("Done.")


