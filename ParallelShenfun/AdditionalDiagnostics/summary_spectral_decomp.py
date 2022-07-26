#!/usr/bin/env python

# post-processing script to compute the spectral energy fluxes and
# transfers from multiple runs of the qgmhd_shenfun.py driver script 
# to plot time averaged and azimuthal averaged version of them as a 
# function of k (wavenumbers) and in multiple cases of M2.
#
# Current outputs: T(k) and -\int T(k) components, viscous and sums
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
#
# As long as nrows*ncols = len(mag_values), you can pick whatever
# configuration of rows and columns you want the panels arranged in.
# The code will go left to right and top to bottom in the order of 
# provided mag_values. This code is however restricted to plotting 
# more than one case. For a single run, consider scripts without 
# the 'summary' prefix.

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
[t_begin,t_end] = [100,150]

# linestyles
color_index = ['k','r','g','b']
lines_index = [(0,()), (0,(5,5)), (0,(3,1,1,1)), (0,(1,1))]
#lines_index = ['-','--','-.',':']
lw = 2.5 

# set the directories you want to read from
case_prefix = "turbB0-F2-0"

# Azimuthal avg: width of annulus in spectral space
dbin = 1.0

# normalize the final curves to be bounded between [-1,1] (look at shape)
normalize = False

# list of directory suffixes corresponding to M2 values 
#mag_values = ["hydro","1em6","1em5","1em4","1em3","1em2","1em1","1em0"]
mag_values = ["hydro","1em6","1em4","1em2"]

# figure panel config
# fixed fig size..
figsize=(4.15,4)

# fig format (eps,png,both)
save_format = "png"

# init dir arrays
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

# read first diag file for invariant quantities
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
V2  = FunctionSpace(N[1], 'F', dtype='D', domain=(0, L[1]))
T   = TensorProductSpace(MPI.COMM_SELF, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
TV  = VectorSpace(T)

X  = T.local_mesh(True)
K = np.array(T.local_wavenumbers(True,True))
KP = np.fft.fftshift(K)

kx = K[0][:,0]
ky = K[1][0,:]

# maximum wavenumbers
nxh = int(N[0]/2)
nyh = int(N[1]/2)    

#kx = np.fft.fftfreq(N[0], d=L[0]/(2.*np.pi))*N[0]
#ky = np.fft.fftfreq(N[1], d=L[1]/(2.*np.pi))*N[1]
kxp = np.fft.fftshift(kx)
kyp = np.fft.fftshift(ky)

# Compute a radial coordinate
[kxx, kyy] = np.meshgrid(kx,ky)
[kxxp,kyyp]= np.meshgrid(kxp,kyp)
R          = np.sqrt(kxx**2 + kyy**2)
RP         = np.sqrt(kxxp**2 + kyyp**2)

np.seterr(divide='ignore', invalid='ignore')
iK = 1j*K
#iK     = np.zeros((2,N[1],N[0]), dtype=complex)
#iK[0]  = 1j*kxx
#iK[1]  = 1j*kyy

if F2 == 0:
    print("F is zero")
    K2oK2F2  = 1
else:
    K2 = K[0]**2+K[1]**2
    K2oK2F2  = np.where((K2 + F2) == 0.0, 0.0, K2/(K2+F2)).astype(float)
    print("F is non-zero")

# Azimuthal avg
kbin = 2.*np.pi/L[0]*np.arange(0., nxh+1, dbin)

# fluxes
HEadv_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])
HElor_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])
HEvis_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEadv_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEvis_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])
TEdyn_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])
TEvis_vecf = np.zeros([len(mag_values),len(tplot),len(kbin)])

#transfers
HEadv_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])
HElor_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])
HEvis_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEadv_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])
MEvis_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])
TEdyn_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])
TEvis_vect = np.zeros([len(mag_values),len(tplot),len(kbin)])

#constants for flux computations
Lo2p2 = (L[0]*L[1]/(4*np.pi**2))

## Main loop to create averaged spectra..
for index in range(len(mag_values)):

    print("Now computing M2 = %1e" % M2_vector[index])
    M2 = M2_vector[index]
    if M2 == 0: M2 = 1e-6; # when M is zero, plot something...

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

        # define transfers (2D)
        HE_advect  = +Lo2p2*(psi_hat*np.conjugate(fftn(inner(u,gradq)))).real
        HE_lorentz = -M2*Lo2p2*(psi_hat*np.conjugate(fftn(inner(b,gradj)))).real
        HE_viscous = -2*K2oK2F2/Lo2p2*(q_hat*np.conjugate(q_hat)/Re).real
        ME_advect  = +M2*Lo2p2*(j_hat*np.conjugate(fftn(inner(u,gradA)-B0*upert[1]))).real
        ME_viscous = -M2*2/Lo2p2*(j_hat*np.conjugate(j_hat)/Rm).real
        if M2_vector[index] == 0:
            TE_dynamic = HE_advect 
            TE_viscous = HE_viscous 
        else:
            TE_dynamic = HE_advect + HE_lorentz + ME_advect
            TE_viscous = HE_viscous + ME_viscous
        
        HE_advect_bin  =  np.zeros(kbin.shape)
        HE_lorentz_bin =  np.zeros(kbin.shape)
        HE_viscous_bin =  np.zeros(kbin.shape) 
        ME_advect_bin  =  np.zeros(kbin.shape)
        ME_viscous_bin =  np.zeros(kbin.shape)
        TE_dynamic_bin =  np.zeros(kbin.shape)
        TE_viscous_bin =  np.zeros(kbin.shape)

        # compute azimuthal averaging and multiply by 2pik (polar jacobian)
        for jj in range(0, len(kbin)-1):
            mask = np.logical_and(RP >= kbin[jj], RP < kbin[jj+1]).astype(int)
            #theta_int = 2*np.pi*kbin[jj]
            theta_int = 1 #dbin # np.pi*(kbin[jj+1]**2-kbin[jj]**2)
            HE_advect_bin[jj]  = ndimage.sum(np.fft.fftshift(HE_advect).real, mask)*theta_int
            HE_lorentz_bin[jj] = ndimage.sum(np.fft.fftshift(HE_lorentz).real, mask)*theta_int
            HE_viscous_bin[jj] = ndimage.sum(np.fft.fftshift(HE_viscous).real, mask)*theta_int
            ME_advect_bin[jj]  = ndimage.sum(np.fft.fftshift(ME_advect).real, mask)*theta_int
            ME_viscous_bin[jj] = ndimage.sum(np.fft.fftshift(ME_viscous).real, mask)*theta_int
            TE_dynamic_bin[jj] = ndimage.sum(np.fft.fftshift(TE_dynamic).real, mask)*theta_int
            TE_viscous_bin[jj] = ndimage.sum(np.fft.fftshift(TE_viscous).real, mask)*theta_int

        # store transfers for temporal averaging
        HEadv_vect[index,ii,:] = HE_advect_bin 
        HElor_vect[index,ii,:] = HE_lorentz_bin
        MEadv_vect[index,ii,:] = ME_advect_bin
        HEvis_vect[index,ii,:] = HE_viscous_bin
        MEvis_vect[index,ii,:] = ME_viscous_bin
        TEdyn_vect[index,ii,:] = TE_dynamic_bin
        TEvis_vect[index,ii,:] = TE_viscous_bin

        # for Pi(k) instead of T(k): computes -\int_0^k T(k) -  i.e. the fluxes
        HE_advect_bin   =  -np.cumsum(HE_advect_bin)
        HE_lorentz_bin  =  -np.cumsum(HE_lorentz_bin)
        HE_viscous_bin  =  -np.cumsum(HE_viscous_bin)
        ME_advect_bin   =  -np.cumsum(ME_advect_bin)
        ME_viscous_bin  =  -np.cumsum(ME_viscous_bin)
        TE_dynamic_bin  =  -np.cumsum(TE_dynamic_bin)
        TE_viscous_bin  =  -np.cumsum(TE_viscous_bin)

        # store fluxes for temporal averaging
        HEadv_vecf[index,ii,:] = HE_advect_bin 
        HElor_vecf[index,ii,:] = HE_lorentz_bin
        MEadv_vecf[index,ii,:] = ME_advect_bin
        HEvis_vecf[index,ii,:] = HE_viscous_bin
        MEvis_vecf[index,ii,:] = ME_viscous_bin
        TEdyn_vecf[index,ii,:] = TE_dynamic_bin
        TEvis_vecf[index,ii,:] = TE_viscous_bin


print("Finished Creating Datasets, now averaging...")

### one loop iteration for each value of M and in each plot all fields..
for index in range(len(mag_values)):

    fig_dyn, axesdyn = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)
    fig_vis, axesvis = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)

    HEvis = np.mean(HEvis_vect[index,first:last+1,:],axis=0)
    MEvis = np.mean(MEvis_vect[index,first:last+1,:],axis=0)
    HEadv = np.mean(HEadv_vect[index,first:last+1,:],axis=0)
    HElor = np.mean(HElor_vect[index,first:last+1,:],axis=0)
    MEadv = np.mean(MEadv_vect[index,first:last+1,:],axis=0)
    TEdyn = np.mean(TEdyn_vect[index,first:last+1,:],axis=0)
    TEvis = np.mean(TEvis_vect[index,first:last+1,:],axis=0)

    maxHEvis = np.max(abs(HEvis))
    maxMEvis = np.max(abs(MEvis))
    maxHEadv = np.max(abs(HEadv))
    maxHElor = np.max(abs(HElor))
    maxMEadv = np.max(abs(MEadv))
    maxTEdyn = np.max(abs(TEdyn))
    maxTEvis = np.max(abs(TEvis))

    if normalize: 
        if maxHEvis > 0: HEvis = HEvis/maxHEvis;
        if maxMEvis > 0: MEvis = MEvis/maxMEvis;
        if maxHEadv > 0: HEadv = HEadv/maxHEadv;
        if maxHElor > 0: HElor = HElor/maxHElor;
        if maxMEadv > 0: MEadv = MEadv/maxMEadv;
        if maxTEdyn > 0: TEdyn = TEdyn/maxTEdyn;
        if maxTEvis > 0: TEvis = TEvis/maxTEvis;

    print("For M2 = ",M2_vector[index],": (transfer)")
    print("Max HEvis: ",maxHEvis)
    print("Max MEvis: ",maxMEvis)
    print("Max HEadv: ",maxHEadv)
    print("Max HElor: ",maxHElor)
    print("Max MEadv: ",maxMEadv)
    print("Max TEdyn: ",maxTEdyn)
    print("Max TEvis: ",maxTEvis)

    axesdyn.plot(kbin, TEdyn, lw=lw, linestyle = lines_index[0], color='k', label=r'$T$')
    axesdyn.plot(kbin, HEadv, lw=lw, linestyle = lines_index[1], color='r', label=r'$T_{q}$')
    axesdyn.plot(kbin, HElor, lw=lw, linestyle = lines_index[2], color='g', label=r'$T_{L}$')
    axesdyn.plot(kbin, MEadv, lw=lw, linestyle = lines_index[3], color='b', label=r'$T_{A}$')
    axesvis.plot(kbin, TEvis, lw=lw, linestyle = lines_index[0], color='k', label=r'$D$')
    axesvis.plot(kbin, HEvis, lw=lw, linestyle = lines_index[1], color='r', label=r'$D_H$')
    axesvis.plot(kbin, MEvis, lw=lw, linestyle = lines_index[3], color='b', label=r'$D_M$')

    dynmax = max(maxHEadv,maxMEadv,maxHElor,maxTEdyn)*1.1
    axesdyn.set_ylim(-dynmax,dynmax)

    for axes in [axesdyn,axesvis]:
        axes.set_xscale('log')
        axes.set_xlabel("wavenumber (k)");
        axes.grid(True)
        axes.tick_params(axis='x')
        axes.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axes.legend(loc='lower right')
        #axes.set_title(r"$M = $"+fmt(np.sqrt(M2_vector[index])))

    for fig in [fig_vis,fig_dyn]:
        fig.tight_layout()

    fig_dyn.savefig("qgmhd_TEdyn_trans_%1.0e.png" % M2_vector[index])
    fig_vis.savefig("qgmhd_TEvis_trans_%1.0e.png" % M2_vector[index])

# now do the fluxes

### one loop iteration for each value of M and in each plot all fields..
for index in range(len(mag_values)):

    fig_dyn, axesdyn = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)
    fig_vis, axesvis = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=figsize)

    HEvis = np.mean(HEvis_vecf[index,first:last+1,:],axis=0)
    MEvis = np.mean(MEvis_vecf[index,first:last+1,:],axis=0)
    HEadv = np.mean(HEadv_vecf[index,first:last+1,:],axis=0)
    HElor = np.mean(HElor_vecf[index,first:last+1,:],axis=0)
    MEadv = np.mean(MEadv_vecf[index,first:last+1,:],axis=0)
    TEdyn = np.mean(TEdyn_vecf[index,first:last+1,:],axis=0)
    TEvis = np.mean(TEvis_vecf[index,first:last+1,:],axis=0)

    maxHEvis = np.max(abs(HEvis))
    maxMEvis = np.max(abs(MEvis))
    maxHEadv = np.max(abs(HEadv))
    maxHElor = np.max(abs(HElor))
    maxMEadv = np.max(abs(MEadv))
    maxTEdyn = np.max(abs(TEdyn))
    maxTEvis = np.max(abs(TEvis))

    if normalize: 
        if maxHEvis > 0: HEvis = HEvis/maxHEvis;
        if maxMEvis > 0: MEvis = MEvis/maxMEvis;
        if maxHEadv > 0: HEadv = HEadv/maxHEadv;
        if maxHElor > 0: HElor = HElor/maxHElor;
        if maxMEadv > 0: MEadv = MEadv/maxMEadv;
        if maxTEdyn > 0: TEdyn = TEdyn/maxTEdyn;
        if maxTEvis > 0: TEvis = TEvis/maxTEvis;

    print("For M2 = ",M2_vector[index],": (transfer)")
    print("Max HEvis: ",maxHEvis)
    print("Max MEvis: ",maxMEvis)
    print("Max HEadv: ",maxHEadv)
    print("Max HElor: ",maxHElor)
    print("Max MEadv: ",maxMEadv)
    print("Max TEdyn: ",maxTEdyn)
    print("Max TEvis: ",maxTEvis)

    axesdyn.plot(kbin, TEdyn, lw=lw, linestyle = lines_index[0], color='k', label=r'$\Pi$') 
    axesdyn.plot(kbin, HEadv, lw=lw, linestyle = lines_index[1], color='r', label=r'$\Pi_q$') 
    axesdyn.plot(kbin, HElor, lw=lw, linestyle = lines_index[2], color='g', label=r'$\Pi_L$') 
    axesdyn.plot(kbin, MEadv, lw=lw, linestyle = lines_index[3], color='b', label=r'$\Pi_A$') 
    axesvis.plot(kbin, TEvis, lw=lw, linestyle = lines_index[0], color='k', label=r'$\epsilon$') 
    axesvis.plot(kbin, HEvis, lw=lw, linestyle = lines_index[1], color='r', label=r'$\epsilon_H$') 
    axesvis.plot(kbin, MEvis, lw=lw, linestyle = lines_index[3], color='b', label=r'$\epsilon_M$') 

    dynmax = max(maxHEadv,maxMEadv,maxHElor,maxTEdyn)*1.1
    axesdyn.set_ylim(-dynmax,dynmax)

    for axes in [axesdyn,axesvis]:
        axes.set_xscale('log')
        axes.set_xlabel("wavenumber (k)");
        axes.grid(True)
        axes.tick_params(axis='x')
        axes.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axes.legend(loc='lower right')
        #axes.set_title(r"$M = $"+fmt(np.sqrt(M2_vector[index])))

    for fig in [fig_vis,fig_dyn]:
        fig.tight_layout()

    fig_dyn.savefig("qgmhd_TEdyn_flux_%1.0e.png" % M2_vector[index])
    fig_vis.savefig("qgmhd_TEvis_flux_%1.0e.png" % M2_vector[index])

print("Done.")


