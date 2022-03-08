#!/usr/bin/env python

# post-processing script to plot the spectral slopes
# from the output of the qgmhd_shenfun.py driver script.
#
# Current outputs: E(k), E_P, E_V, E_M
#
# Options to make a movie or just animate

# imports
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import h5py
import subprocess
from shenfun import *
from scipy import ndimage
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
        return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)
    def zeros(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align(nzeros(N, dtype=dtype), bytes)
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

# Flag to consider the full fields (q = \bar q + q') or perturbations (q')
full_field = False

# make movie
movie = True
movie_name = 'qgmhd_slope.mp4'

# time bounds for temporal averaging
[t_begin,t_end] = [100,300]; t_snap = t_end # snap of PV with avg slopes

k_bound = [5,10] # sets the wavenumber bounds for slope computations
lims_on = False # turns on/off y-axis limits on spectral slope plots
lims = [1e-1, 1e11] # sets values of those limits

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
Uj = file2['Uj'][...]
assert(full_field*Uj == 0), "Avoid computing the full spectrum for the jet, Streamfunction is not periodic."
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

x = np.array(f[var1 + '/domain/x0'][()])       # bounds in 0-axis
y = np.array(f[var1 + '/domain/x1'][()])       # bounds in 1-axis

L = (x[-1], y[-1])

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

plt.figure(figsize=(9,9))

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

KE_vec = np.zeros([len(tplot),len(kbin)])
ME_vec = np.zeros([len(tplot),len(kbin)])
PE_vec = np.zeros([len(tplot),len(kbin)])
TE_vec = np.zeros([len(tplot),len(kbin)])

for ii in range(len(tplot)):

    t = tplot[ii]

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

    PEspectrum = scale*F2*np.fft.fftshift(abs(psi_hat)**2)
    KEspectrum = scale*np.fft.fftshift(abs(u_hat)**2+abs(v_hat)**2)
    MEspectrum = scale*M2*np.fft.fftshift(abs(b1_hat)**2+abs(b2_hat)**2)
    TEspectrum = KEspectrum + PEspectrum + MEspectrum

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
    
    I1 = np.array( np.where(((kbin > k_bound[0]) & (kbin < k_bound[1]))) )
    I1 = I1.ravel()
    if len(I1) > 2:
        p1 = np.polyfit(np.log10(kbin[I1]), np.log10(KEbin[I1]), 1)
        p2 = np.polyfit(np.log10(kbin[I1]), np.log10(MEbin[I1]), 1)
        p3 = np.polyfit(np.log10(kbin[I1]), np.log10(TEbin[I1]), 1)
        p4 = np.polyfit(np.log10(kbin[I1]), np.log10(PEbin[I1]), 1)

    KE_vec[ii] = KEbin
    ME_vec[ii] = MEbin
    PE_vec[ii] = PEbin
    TE_vec[ii] = TEbin
    
    print("time=%3.3f, KE=%10.10f, ME=%10.10f, TE=%10.10f" % (t, p1[0], p2[0], p3[0]))
    plt.clf()
    plt.subplot(2,2,1)
    plt.loglog(kbin[1:], TEmult*TEbin[1:], 'ob')
    plt.loglog(kbin[1:], TEmult*pow(10,np.polyval(p3,np.log10(kbin[1:]))), '-k', label=r'$\hat E$ Slope: % 2.2f' % p3[0])
    plt.loglog(kbin[I1[0]], TEmult[I1[0]]*pow(10,np.polyval(p3,np.log10(kbin[I1[0]]))), 'or')
    plt.loglog(kbin[I1[-1]], TEmult[I1[-1]]*pow(10,np.polyval(p3,np.log10(kbin[I1[-1]]))), 'or')
    if lims_on: plt.ylim(lims);
    plt.grid(True)
    plt.xlabel("k"); plt.ylabel(r'$\hat E$'); plt.legend(loc='best')
    plt.title('Mean TE Spectrum at t = %4.2f' % t)

    plt.subplot(2,2,2)
    plt.loglog(kbin[1:], PEmult*PEbin[1:], 'ob')
    plt.loglog(kbin[1:], PEmult*pow(10,np.polyval(p4,np.log10(kbin[1:]))), '-k', label=r'$\hat E_P$ Slope: % 2.2f' % p4[0])
    plt.loglog(kbin[I1[0]], PEmult[I1[0]]*pow(10,np.polyval(p4,np.log10(kbin[I1[0]]))), 'or')
    plt.loglog(kbin[I1[-1]], PEmult[I1[-1]]*pow(10,np.polyval(p4,np.log10(kbin[I1[-1]]))), 'or')
    if lims_on: plt.ylim(lims);
    plt.grid(True)
    plt.xlabel("k"); plt.ylabel(r'$\hat E_P$'); plt.legend(loc='best')
    plt.title('Mean PE Spectrum at t = %4.2f' % t)

    plt.subplot(2,2,3)
    plt.loglog(kbin[1:], KEmult*KEbin[1:], 'ob')
    plt.loglog(kbin[1:], KEmult*pow(10,np.polyval(p1,np.log10(kbin[1:]))), '-k', label=r'$\hat E_V$ Slope: % 2.2f' % p1[0])
    plt.loglog(kbin[I1[0]], KEmult[I1[0]]*pow(10,np.polyval(p1,np.log10(kbin[I1[0]]))), 'or')
    plt.loglog(kbin[I1[-1]], KEmult[I1[-1]]*pow(10,np.polyval(p1,np.log10(kbin[I1[-1]]))), 'or')
    if lims_on: plt.ylim(lims);
    plt.grid(True)
    plt.xlabel("k"); plt.ylabel(r'$\hat E_V$'); plt.legend(loc='best')
    plt.title('Mean KE Spectrum at t = %4.2f' % t)

    plt.subplot(2,2,4)
    plt.loglog(kbin[1:], MEmult*MEbin[1:], 'ob')
    plt.loglog(kbin[1:], MEmult*pow(10,np.polyval(p2,np.log10(kbin[1:]))), '-k', label=r'$\hat E_M$ Slope: % 2.2f' % p2[0])
    plt.loglog(kbin[I1[0]], MEmult[I1[0]]*pow(10,np.polyval(p2,np.log10(kbin[I1[0]]))), 'or')
    plt.loglog(kbin[I1[-1]], MEmult[I1[-1]]*pow(10,np.polyval(p2,np.log10(kbin[I1[-1]]))), 'or')
    if lims_on: plt.ylim(lims);
    plt.grid(True)
    plt.xlabel("k");plt.ylabel(r'$\hat E_M$'); plt.legend(loc='best')
    plt.title('Mean ME Spectrum at t = %4.2f' % t)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    if movie:
        plt.savefig('frame_{0:04d}.png'.format(ii), dpi=200)

if movie:
    merge_to_mp4('frame_%04d.png', movie_name)

#plt.show()

plt.figure(figsize=(7,7))

# temporal averaging

first = int(np.where(np.array(tplot)==t_begin)[0])
last = int(np.where(np.array(tplot)==t_end)[0])

#Tkavg = np.mean(Tkavg,axis=0)
KEbin = np.mean(KE_vec[first:last,:],axis=0)
MEbin = np.mean(ME_vec[first:last,:],axis=0)
PEbin = np.mean(PE_vec[first:last,:],axis=0)
TEbin = np.mean(TE_vec[first:last,:],axis=0)

I1 = np.array( np.where(((kbin > k_bound[0]) & (kbin < k_bound[1]))) )
I1 = I1.ravel()
if len(I1) > 2:
    p1 = np.polyfit(np.log10(kbin[I1]), np.log10(KEbin[I1]), 1)
    p2 = np.polyfit(np.log10(kbin[I1]), np.log10(MEbin[I1]), 1)
    p3 = np.polyfit(np.log10(kbin[I1]), np.log10(TEbin[I1]), 1)
    p4 = np.polyfit(np.log10(kbin[I1]), np.log10(PEbin[I1]), 1)

print('Slope: ', p3[0])

jj = int(np.where(np.array(tplot)==t_snap)[0])

plt.clf()
plt.subplot(2,2,1)
plt.loglog(kbin[1:], TEmult*TEbin[1:], 'ob')
plt.loglog(kbin[1:], TEmult*pow(10,np.polyval(p3,np.log10(kbin[1:]))), '-k', label=r'$\hat E$ Slope: % 2.2f' % p3[0])
plt.loglog(kbin[I1[0]], TEmult[I1[0]]*pow(10,np.polyval(p3,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], TEmult[I1[-1]]*pow(10,np.polyval(p3,np.log10(kbin[I1[-1]]))), 'or')
if lims_on: plt.ylim(lims);
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$\hat E$')
plt.legend(loc='best');
plt.title('Temporal Mean TE Spectrum' % t)

plt.subplot(2,2,2)
plt.loglog(kbin[1:], PEmult*PEbin[1:], 'ob')
plt.loglog(kbin[1:], PEmult*pow(10,np.polyval(p4,np.log10(kbin[1:]))), '-k', label=r'$\hat E_P$ Slope: % 2.2f' % p4[0])
plt.loglog(kbin[I1[0]], PEmult[I1[0]]*pow(10,np.polyval(p4,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], PEmult[I1[-1]]*pow(10,np.polyval(p4,np.log10(kbin[I1[-1]]))), 'or')
if lims_on: plt.ylim(lims);
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$\hat E_P$')
plt.legend(loc='best');
plt.title('Temporal Mean PE Spectrum' % t)

plt.subplot(2,2,3)
plt.loglog(kbin[1:], KEmult*KEbin[1:], 'ob')
plt.loglog(kbin[1:], KEmult*pow(10,np.polyval(p1,np.log10(kbin[1:]))), '-k', label=r'$\hat E_V$ Slope: % 2.2f' % p1[0])
plt.loglog(kbin[I1[0]], KEmult[I1[0]]*pow(10,np.polyval(p1,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], KEmult[I1[-1]]*pow(10,np.polyval(p1,np.log10(kbin[I1[-1]]))), 'or')
if lims_on: plt.ylim(lims);
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$\hat E_V$')
plt.legend(loc='best');
plt.title('Temporal Mean KE Spectrum' % t)

plt.subplot(2,2,4)
plt.loglog(kbin[1:], MEmult*MEbin[1:], 'ob')
plt.loglog(kbin[1:], MEmult*pow(10,np.polyval(p2,np.log10(kbin[1:]))), '-k', label=r'$\hat E_M$ Slope: % 2.2f' % p2[0])
plt.loglog(kbin[I1[0]], MEmult[I1[0]]*pow(10,np.polyval(p2,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], MEmult[I1[-1]]*pow(10,np.polyval(p2,np.log10(kbin[I1[-1]]))), 'or')
if lims_on: plt.ylim(lims);
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$\hat E_M$')
plt.legend(loc='best');
plt.title('Temporal Mean ME Spectrum' % t)

plt.tight_layout()
#plt.show()

file_name_png = "qgmhd_slope_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

