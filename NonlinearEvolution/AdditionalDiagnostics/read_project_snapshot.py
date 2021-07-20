import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import subprocess

from scipy import ndimage
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

M = 7
N = (2**M, 2**M)

folder           = 'output-qgmhd/'
file_name = folder + "qgmhd_Nx{}_diagnostics.h5".format(N[0])
filenc    = folder + 'qgmhd_Nx{}_variables.nc'.format(N[0])

f = Dataset(filenc, 'r', format='NETCDF4')

#print(f)
#print(f.variables.keys())
#tt = f.variables['Time']

tp = f.variables['TimePlot']
x  = f.variables['x']
y  = f.variables['y']

Lx=max(np.abs(x))/(1-0.5/np.size(x))
Ly=max(np.abs(y))/(1-0.5/np.size(y))

Q = f.variables['PV']
#A = f.variables['A']
j = f.variables['j']
u = f.variables['u']
v = f.variables['v']
b1= f.variables['b1']
b2= f.variables['b2']

plt.figure(figsize=(10,8))

# maximum wavenumbers
nxh = int(N[0]/2)
nyh = int(N[1]/2)    

kx = np.fft.fftfreq(N[0], d=Lx/(2.*np.pi))*N[0]
ky = np.fft.fftfreq(N[1], d=Ly/(2.*np.pi))*N[1]
kxp = np.fft.fftshift(kx)
kyp = np.fft.fftshift(ky)

# Compute a radial coordiante
[kxxp, kyyp] = np.meshgrid(kxp,kyp)
R = np.sqrt(kxxp**2 + kyyp**2)

# Azimuthal avg
dbin = 1.0
kbin = 2.*np.pi/Lx*np.arange(0., nxh+1, dbin)
KEbin = np.zeros(kbin.shape)
MEbin = np.zeros(kbin.shape)
TEbin = np.zeros(kbin.shape)

np.seterr(divide='ignore')

#ii = len(tp)-1
t_find = 10
ii = int(np.where(np.array(tp)==t_find)[0])
t = tp[ii]

u_hat = fftn(u[ii,:,:])
v_hat = fftn(v[ii,:,:])
b1_hat = fftn(b1[ii,:,:])
b2_hat = fftn(b2[ii,:,:])

KEspectrum = np.fft.fftshift(abs(u_hat)**2+abs(v_hat)**2)
MEspectrum = np.fft.fftshift(abs(b1_hat)**2+abs(b2_hat)**2)
TEspectrum = KEspectrum + MEspectrum
 
for jj in range(0, len(kbin)-1):
    mask = np.logical_and(R > kbin[jj], R < kbin[jj+1]).astype(int)
    KEbin[jj] = ndimage.mean(KEspectrum, mask)*np.pi*(kbin[jj+1]**2-kbin[jj]**2)
    MEbin[jj] = ndimage.mean(MEspectrum, mask)*np.pi*(kbin[jj+1]**2-kbin[jj]**2)
    TEbin[jj] = ndimage.mean(TEspectrum, mask)*np.pi*(kbin[jj+1]**2-kbin[jj]**2)

k_bound = [1, 10]
I1 = np.array( np.where(((kbin > k_bound[0]) & (kbin < k_bound[1]))) )
I1 = I1.ravel()

if len(I1) > 2:
    p1 = np.polyfit(np.log10(kbin[I1]), np.log10(KEbin[I1]), 1)
    p2 = np.polyfit(np.log10(kbin[I1]), np.log10(MEbin[I1]), 1)
    p3 = np.polyfit(np.log10(kbin[I1]), np.log10(TEbin[I1]), 1)

print('Slope: ', p3[0])

plt.clf()
plt.subplot(2,2,1)
plt.pcolormesh(x,y,Q[ii], cmap='seismic', shading = 'auto')
plt.title('PV at t = %4.2f' % t)
plt.xlabel("x")
plt.ylabel("y")
plt.clim([-np.max(abs(Q[ii])),np.max(abs(Q[ii]))])
plt.colorbar()

##################
KEmult = kbin[1:]**(5/3)
MEmult = KEmult
TEmult = kbin[1:]**(3/2)

lims = [1e0, 1e12]
##################

plt.subplot(2,2,2)
plt.loglog(kbin[1:], TEmult*TEbin[1:], 'ob')
plt.loglog(kbin[1:], TEmult*pow(10,np.polyval(p3,np.log10(kbin[1:]))), '-k', label=r'$\hat E$ Slope: % 4.4f' % p3[0])
plt.loglog(kbin[I1[0]], TEmult[I1[0]]*pow(10,np.polyval(p3,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], TEmult[I1[-1]]*pow(10,np.polyval(p3,np.log10(kbin[I1[-1]]))), 'or')
#plt.loglog(kxp[nxh:], spectrum[nxh,nxh:],'xr', label='Slice')
plt.ylim(lims)
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$k^{3/2}\hat E$')
plt.legend(loc='best');
plt.title('Mean TE Spectrum at t = %4.2f' % t)

plt.subplot(2,2,3)
plt.loglog(kbin[1:], KEmult*KEbin[1:], 'ob')
plt.loglog(kbin[1:], KEmult*pow(10,np.polyval(p1,np.log10(kbin[1:]))), '-k', label=r'$\hat E_V$ Slope: % 4.4f' % p1[0])
plt.loglog(kbin[I1[0]], KEmult[I1[0]]*pow(10,np.polyval(p1,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], KEmult[I1[-1]]*pow(10,np.polyval(p1,np.log10(kbin[I1[-1]]))), 'or')
#plt.loglog(kxp[nxh:], spectrum[nxh,nxh:],'xr', label='Slice')
plt.ylim(lims)
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$k^{5/3}\hat E_V$')
plt.legend(loc='best');
plt.title('Mean KE Spectrum at t = %4.2f' % t)

plt.subplot(2,2,4)
plt.loglog(kbin[1:], MEmult*MEbin[1:], 'ob')
plt.loglog(kbin[1:], MEmult*pow(10,np.polyval(p2,np.log10(kbin[1:]))), '-k', label=r'$\hat E_M$ Slope: % 4.4f' % p2[0])
plt.loglog(kbin[I1[0]], MEmult[I1[0]]*pow(10,np.polyval(p2,np.log10(kbin[I1[0]]))), 'or')
plt.loglog(kbin[I1[-1]], MEmult[I1[-1]]*pow(10,np.polyval(p2,np.log10(kbin[I1[-1]]))), 'or')
#plt.loglog(kxp[nxh:], spectrum[nxh,nxh:],'xr', label='Slice')
plt.ylim(lims)
plt.grid(True)
plt.xlabel("k")
plt.ylabel(r'$k^{5/3}\hat E_M$')
plt.legend(loc='best');
plt.title('Mean ME Spectrum at t = %4.2f' % t)

plt.tight_layout()
plt.draw()
#plt.savefig("spectrum.png")
plt.show()



