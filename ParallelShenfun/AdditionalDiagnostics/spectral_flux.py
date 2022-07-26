#!/usr/bin/env python

# post-processing script to plot the spectral fluxes
# from the output of the qgmhd_shenfun.py driver script.
#
# Current outputs: T(k), -\int T(k) dk, viscous and full equation
#
# Options to make a movie or just animate

# imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import subprocess
import time
from scipy import ndimage, integrate
from shenfun import *
from library.operators import inner
from library.data_output import merge_to_mp4

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

# set resolution to pick out correct files
N = [1024]*2

movie = True
movie_name = 'qgmhd_fluxes.mp4'

# time bounds for temporal averaging
[t_begin,t_end] = [100,150] 

folder           = 'output-qgmhd/'
file_name = folder + "qgmhd_Nx{}_diagnostics.h5".format(N[0])
filenc    = folder + 'qgmhd_Nx{}_fields.h5'.format(N[0])

f = h5py.File(filenc, "r")
file2 = h5py.File(file_name, "r")

times  = file2['times']
twrite = file2['twrite']
domain = file2['domain']
M2 = file2['M2'][...]
F2 = file2['F2'][...]
B0 = file2['B0'][...]
Uj = file2['Uj'][...]
Re = file2['Re'][...]
Rm = file2['Rm'][...]

t0,tf,dt = times[0], times[1], times[2]
tt       = np.arange(t0, tf, dt)
ntplot   = int(twrite[...]/times[2])
tplot    = tt[0::ntplot]

# Get information from file
keys = list(f.keys()) # list variables

# Which variable to view
var1 = keys[0]

x0 = np.array(f[var1 + '/domain/x0'][()])       # bounds in 0-axis
x1 = np.array(f[var1 + '/domain/x1'][()])       # bounds in 1-axis

print('Lengths of domain:')
print('      x0 in [%6.4f, %6.4f]' % (x0[0], x0[-1]))
print('      x1 in [%6.4f, %6.4f]' % (x1[0], x1[-1]))

L = (x0[-1], x1[-1])

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

# Compute a radial coordiante
[kxx, kyy] = np.meshgrid(kx,ky)
[kxxp,kyyp]= np.meshgrid(kxp,kyp)
R          = np.sqrt(kxx**2 + kyy**2)
RP         = np.sqrt(kxxp**2 + kyyp**2)

np.seterr(divide='ignore')
iK = 1j*K
#iK     = np.zeros((2,N[1],N[0]), dtype=complex)
#iK[0]  = 1j*kxx
#iK[1]  = 1j*kyy

if F2 == 0:
    print("F is zero")
    K2oK2F2  = 1
else:
    K2oK2F2  = np.where((K**2 + F2) == 0.0, 0.0, K**2/(K**2+F2)).astype(float)
    print("F is non-zero")

# Azimuthal avg
dbin = 1
kbin = 2.*np.pi/L[0]*np.arange(0., nxh+1, dbin)

HEadv_vec = np.zeros([len(tplot),len(kbin)])
HElor_vec = np.zeros([len(tplot),len(kbin)])
HEvis_vec = np.zeros([len(tplot),len(kbin)])
MEadv_vec = np.zeros([len(tplot),len(kbin)])
MEvis_vec = np.zeros([len(tplot),len(kbin)])
TEtot_vec = np.zeros([len(tplot),len(kbin)])

## Plotting Initialization

plt.ion()
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
fig.subplots_adjust(wspace=0.4, bottom=0.15)

p1, = ax[0].plot(kbin[1:], HEadv_vec[0,1:], '-r', label=r'$-\int T_{q}(k) dk$')
p2, = ax[0].plot(kbin[1:], HElor_vec[0,1:], '-b', label=r'$-\int T_{L}(k) dk$')
p3, = ax[0].plot(kbin[1:], MEadv_vec[0,1:], '-g', label=r'$-\int T_{A}(k) dk$')
p4, = ax[0].plot(kbin[1:], TEtot_vec[0,1:], '-k', label=r'$-\int T(k) dk$')

twin1 = ax[1].twinx() 
p5, = ax[1].plot(kbin[1:], HEvis_vec[0,1:], '-r', label=r'$2R_e^{-1}\int k^2\hat E_Hdk$')
p6, = twin1.plot(kbin[1:], MEvis_vec[0,1:], '-b', label=r'$2R_m^{-1}\int k^2\hat E_Mdk$')
    
#ax[0].set_xscale('log')
ax[0].set_xlabel("wavenumber (k)")
ax[0].grid(True)
ax[0].tick_params(axis='x')
ax[0].legend(handles=[p1, p2, p3, p4],loc='upper left')

#ax[1].set_xscale('log')
ax[1].set_xlabel("wavenumber (k)")
ax[1].grid(True)
ax[1].tick_params(axis='x')
ax[1].legend(handles=[p5, p6],loc='upper left')
ax[1].set_xlabel("wavenumber (k)")

ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
twin1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

for [axes,plots] in [[ax[1],p5],[twin1,p6]]:
    axes.tick_params(axis='y', colors=plots.get_color())

#constants for flux computations
Lo2p2 = (L[0]*L[1]/(4*np.pi**2))

## Evolve and update plot
for ii in np.arange(0, Nt):

    t  = tplot[ii]

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
    HE_lorentz = -M2*Lo2p2*(psi_hat*np.conjugate(fftn(inner(b,gradj)))).real
    HE_viscous = -2*K2oK2F2/Lo2p2*(q_hat*np.conjugate(q_hat)/Re).real
    ME_advect  = +M2*Lo2p2*(j_hat*np.conjugate(fftn(inner(u,gradA)-B0*upert[1]))).real
    ME_viscous = -M2*2/Lo2p2*(j_hat*np.conjugate(j_hat)/Rm).real
    TE_total   = HE_advect + HE_lorentz + ME_advect

    print("2D sum = ", np.sum(HE_advect))

    HE_advect_bin  =  np.zeros(kbin.shape)
    HE_lorentz_bin =  np.zeros(kbin.shape)
    HE_viscous_bin =  np.zeros(kbin.shape) 
    ME_advect_bin  =  np.zeros(kbin.shape)
    ME_viscous_bin =  np.zeros(kbin.shape)
    TE_total_bin   =  np.zeros(kbin.shape)

    for jj in range(0, len(kbin)-1):
        mask = np.logical_and(RP >= kbin[jj], RP < kbin[jj+1]).astype(int)
        #theta_int = 2*np.pi*kbin[jj]
        #theta_int = np.pi*(kbin[jj+1]**2-kbin[jj]**2)
        theta_int = dbin
        HE_advect_bin[jj] = ndimage.sum(np.fft.fftshift(HE_advect).real, mask)*theta_int
        HE_lorentz_bin[jj] = ndimage.sum(np.fft.fftshift(HE_lorentz).real, mask)*theta_int
        HE_viscous_bin[jj] = ndimage.sum(np.fft.fftshift(HE_viscous).real, mask)*theta_int
        ME_advect_bin[jj] = ndimage.sum(np.fft.fftshift(ME_advect).real, mask)*theta_int
        ME_viscous_bin[jj] = ndimage.sum(np.fft.fftshift(ME_viscous).real, mask)*theta_int
        TE_total_bin[jj] = ndimage.sum(np.fft.fftshift(TE_total).real, mask)*theta_int

    HE_advect_bin  =  -np.cumsum(HE_advect_bin)
    
    print("1D sum = ",-HE_advect_bin[-1])
    print("Differ = ",np.sum(HE_advect)+HE_advect_bin[-1])

    HE_lorentz_bin  =  -np.cumsum(HE_lorentz_bin)
    HE_viscous_bin  =  -np.cumsum(HE_viscous_bin)
    ME_advect_bin  =  -np.cumsum(ME_advect_bin)
    ME_viscous_bin  =  -np.cumsum(ME_viscous_bin)
    TE_total_bin   = -np.cumsum(TE_total_bin)

    # for Pi(k) instead of T(k): computes -\int_0^k T(k)
    #HE_advect_bin  = -integrate.cumtrapz(HE_advect_bin, kbin, initial=0)
    #HE_lorentz_bin = -integrate.cumtrapz(HE_lorentz_bin, kbin, initial=0)
    #HE_viscous_bin = -integrate.cumtrapz(HE_viscous_bin, kbin, initial=0)
    #ME_advect_bin  = -integrate.cumtrapz(ME_advect_bin, kbin, initial=0)
    #ME_viscous_bin = -integrate.cumtrapz(ME_viscous_bin, kbin, initial=0)
    #TE_total_bin   = -integrate.cumtrapz(TE_total_bin, kbin, initial=0)

    # store for temporal averaging
    HEadv_vec[ii,:] = HE_advect_bin 
    HElor_vec[ii,:] = HE_lorentz_bin
    MEadv_vec[ii,:] = ME_advect_bin
    HEvis_vec[ii,:] = HE_viscous_bin
    MEvis_vec[ii,:] = ME_viscous_bin
    TEtot_vec[ii,:] = TE_total_bin

    pltmax1 = max(np.max(abs(HE_advect_bin)),np.max(abs(HE_lorentz_bin)),np.max(abs(ME_advect_bin)),np.max(abs(TE_total_bin)))

    ax[0].set_ylim(-pltmax1,pltmax1)

    ax[1].set_ylim(-np.max(abs(HE_viscous_bin)),np.max(abs(HE_viscous_bin)))
    twin1.set_ylim(-np.max(abs(ME_viscous_bin)),np.max(abs(ME_viscous_bin)))

    p1.set_ydata(HE_advect_bin[1:])
    p2.set_ydata(HE_lorentz_bin[1:])
    p3.set_ydata(ME_advect_bin[1:])
    p4.set_ydata(TE_total_bin[1:])

    p5.set_ydata(HE_viscous_bin[1:])
    p6.set_ydata(ME_viscous_bin[1:]) 

    ax[0].set_title('Component Fluxes at t = %2.0f' % t)
    ax[1].set_title('Viscous Fluxes at t = %2.0f' % t)

    fig.canvas.draw()

    if movie:
        fig.savefig('frame_{0:04d}.png'.format(ii), dpi=200)

    fig.canvas.flush_events()
    time.sleep(0.001)

if movie:
    merge_to_mp4('frame_%04d.png', movie_name)


# temporal averaging

first = int(np.where(np.array(tplot)==t_begin)[0])
last = int(np.where(np.array(tplot)==t_end)[0])

HEvis = np.mean(HEvis_vec[first:last,:],axis=0)
MEvis = np.mean(MEvis_vec[first:last,:],axis=0)
HEadv = np.mean(HEadv_vec[first:last,:],axis=0)
HElor = np.mean(HElor_vec[first:last,:],axis=0)
MEadv = np.mean(MEadv_vec[first:last,:],axis=0)
TEtot = np.mean(TEtot_vec[first:last,:],axis=0)

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
fig.subplots_adjust(wspace=0.4, bottom=0.15)

p1, = ax[0].plot(kbin[1:], HEadv[1:], '-r', label=r'$-\int T_{q}(k) dk$')
p2, = ax[0].plot(kbin[1:], HElor[1:], '-b', label=r'$-\int T_{L}(k) dk$')
p3, = ax[0].plot(kbin[1:], MEadv[1:], '-g', label=r'$-\int T_{A}(k) dk$')
p4, = ax[0].plot(kbin[1:], TEtot[1:], '-k', label=r'$-\int T_(k) dk$')

twin1 = ax[1].twinx() 
p5, = ax[1].plot(kbin[1:], HEvis[1:], '-r', label=r'$2R_e^{-1}\int k^2\hat E_Hdk$')
p6, = twin1.plot(kbin[1:], MEvis[1:], '-b', label=r'$2R_m^{-1}\int k^2\hat E_Mdk$')
    
#ax[0].set_xscale('log')
ax[0].set_xlabel("wavenumber (k)")
ax[0].grid(True)
ax[0].tick_params(axis='x')
ax[0].legend(handles=[p1, p2, p3, p4],loc='upper left')
ax[0].set_title('Avg Component Fluxes')

#ax[1].set_xscale('log')
ax[1].set_xlabel("wavenumber (k)")
ax[1].grid(True)
ax[1].tick_params(axis='x')
ax[1].legend(handles=[p5, p6],loc='upper left')
ax[1].set_xlabel("wavenumber (k)")
ax[1].set_title('Avg Viscous Fluxes')

ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
twin1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

for [axes,plots] in [[ax[1],p5],[twin1,p6]]:
    axes.tick_params(axis='y', colors=plots.get_color())

pltmax1 = max(np.max(abs(HEadv)),np.max(abs(HElor)),np.max(abs(MEadv)),np.max(abs(TEtot)))

ax[0].set_ylim(-pltmax1,pltmax1)

ax[1].set_ylim(-np.max(abs(HEvis)),np.max(abs(HEvis)))
twin1.set_ylim(-np.max(abs(MEvis)),np.max(abs(MEvis)))

fig.canvas.draw()

file_name_png = "qgmhd_flux_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

