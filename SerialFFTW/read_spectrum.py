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


def merge_to_mp4(frame_filenames, movie_name, fps=12):
    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    f_log.close()
    f_err.close()

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
#j = f.variables['j']
u = f.variables['u']
v = f.variables['v']
b1= f.variables['b1']
b2= f.variables['b2']
M2=0.0 # magnetic amplitude

movie = False
movie_name = 'qgmhd_1L_spectrum.mp4'

plt.figure(figsize=(15,6))

# maximum wavenumbers
nxh = int(N[0]/2)
nyh = int(N[1]/2)    

kx = np.fft.fftfreq(N[0], d=Lx/(2.*np.pi))*N[0]
ky = np.fft.fftfreq(N[1], d=Ly/(2.*np.pi))*N[1]
kxp = np.fft.fftshift(kx)
kyp = np.fft.fftshift(ky)

# Compute a radial coordiante
[kxx, kyy] = np.meshgrid(kxp,kyp)
R = np.sqrt(kxx**2 + kyy**2)

# Azimuthal avg
dbin = 1.0
kbin = 2.*np.pi/Lx*np.arange(0., nxh+1, dbin)
Ebin = np.zeros(kbin.shape)

np.seterr(divide='ignore')

for ii in range(len(tp)):

    t = tp[ii]
    u_hat = fftn(u[ii,:,:])
    v_hat = fftn(v[ii,:,:])
    b1_hat = fftn(b1[ii,:,:])
    b2_hat = fftn(b2[ii,:,:])

    spectrum = abs(u_hat**2+v_hat**2 + M2*(b1_hat**2+b2_hat**2))
    spectrum = np.fft.fftshift(spectrum)

    for jj in range(1, len(kbin)-1):
        mask = np.logical_and(R > kbin[jj], R < kbin[jj+1]).astype(int)
        Ebin[jj] = ndimage.mean(spectrum, mask)*np.pi*(kbin[jj+1]**2-kbin[jj]**2)
        
    k_bound = [1, 10]
    I1 = np.array( np.where(((kbin > k_bound[0]) & (kbin < k_bound[1]))) )
    I1 = I1.ravel()
    if len(I1) > 2:
        p1 = np.polyfit(np.log10(kbin[I1]), np.log10(Ebin[I1]), 1)

    plt.clf()
    plt.subplot(1,2,1)
    #plt.pcolormesh((kxp),(kyp),np.log10(spectrum),shading='auto')
    plt.pcolormesh(x,y,Q[ii], cmap = 'seismic',shading = 'gouraud')
    #plt.colorbar()
    plt.title('PV at t = %4.2f' % t)
    plt.clim([-np.max(abs(Q[ii])),np.max(abs(Q[ii]))])
    #plt.axis([0, kxp[-1], 0, kyp[-1]])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('PV at t = %4.2f' % t)
    # Plot rad integrated spectrum    
    plt.subplot(1,2,2)
    plt.xscale('log')
    plt.loglog(kbin[1:], Ebin[1:], 'ob', label='Azimuthal Average')
    plt.loglog(kbin[1:], pow(10,np.polyval(p1,np.log10(kbin[1:]))), '-k', label='Slope: % 4.4f' % p1[0])
    plt.loglog(kbin[I1[0]], pow(10,np.polyval(p1,np.log10(kbin[I1[0]]))), 'or')
    plt.loglog(kbin[I1[-1]], pow(10,np.polyval(p1,np.log10(kbin[I1[-1]]))), 'or')  
    plt.grid(True)
    plt.legend(loc='best');
    plt.title('Mean Spectrum at t = %4.2f' % t)
    plt.ylim([1e-4,1e8])
    plt.xlabel("wavenumber (k)")
    plt.ylabel("log10(PSD)")
    plt.draw()
    plt.pause(0.0001)
    if movie:
        plt.savefig('frame_{0:04d}.png'.format(ii), dpi=200)


if movie:
    merge_to_mp4('frame_%04d.png', movie_name)

#plt.show()



