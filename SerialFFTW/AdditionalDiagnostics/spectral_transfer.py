import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import h5py
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


def merge_to_mp4(frame_filenames, movie_name, fps=12):
    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    f_log.close()
    f_err.close()

M = 9
N = (2**M, 2**M)

folder           = 'output-qgmhd/'
file_name = folder + "qgmhd_Nx{}_diagnostics.h5".format(N[0])
filenc    = folder + 'qgmhd_Nx{}_variables.nc'.format(N[0])

f = Dataset(filenc, 'r', format='NETCDF4')
file2 = h5py.File(file_name, "r")

M2 = file2['M2'][0]
F2 = file2['F2'][0]
Uj = file2['Uj'][0]
Re = file2['Re'][0]
Rm = file2['Rm'][0]
amp = file2['amp'][0]

#print(f)
#print(f.variables.keys())
#tt = f.variables['Time']

tp = f.variables['TimePlot']
x  = f.variables['x']
y  = f.variables['y']

Lx=max(np.abs(x))/(1-0.5/np.size(x))
Ly=max(np.abs(y))/(1-0.5/np.size(y))

Q = f.variables['PV']
A = f.variables['A']
j = f.variables['j']
ud = f.variables['u']
vd = f.variables['v']
b1d= f.variables['b1']
b2d= f.variables['b2']

movie = False
movie_name = 'qgmhd_transfers.mp4'

plt.figure(figsize=(9,4))

# maximum wavenumbers
nxh = int(N[0]/2)
nyh = int(N[1]/2)    

kx = np.fft.fftfreq(N[0], d=Lx/(2.*np.pi))*N[0]
ky = np.fft.fftfreq(N[1], d=Ly/(2.*np.pi))*N[1]
kxp = np.fft.fftshift(kx)
kyp = np.fft.fftshift(ky)

# Compute a radial coordiante
[kxx, kyy] = np.meshgrid(kx,ky)
[kxxp,kyyp]= np.meshgrid(kxp,kyp)
R          = np.sqrt(kxx**2 + kyy**2)
RP         = np.sqrt(kxxp**2 + kyyp**2)

iK     = np.zeros((2,N[1],N[0]), dtype=complex)
iK[0]  = 1j*kxx
iK[1]  = 1j*kyy
K2F2inv  = np.where((R**2 + F2) == 0.0, 0.0, 1.0/(R**2+F2)).astype(float)

# Azimuthal avg
dbin = 1.0
kbin = 2.*np.pi/Lx*np.arange(0., nxh+1, dbin)
KEbin = np.zeros(kbin.shape)
MEbin = np.zeros(kbin.shape)
KEflux = np.zeros(kbin.shape)
MEflux = np.zeros(kbin.shape)

np.seterr(divide='ignore')

#t_find = 10
#ii = int(np.where(np.array(tp)==t_find)[0])
#time_vector = [ii]
time_vector = range(len(tp))

KEadv_vec = np.zeros([len(tp),len(kbin)])
KElor_vec = np.zeros([len(tp),len(kbin)])
KEvis_vec = np.zeros([len(tp),len(kbin)])
MEadv_vec = np.zeros([len(tp),len(kbin)])
MEvis_vec = np.zeros([len(tp),len(kbin)])

for ii in time_vector:

    t  = tp[ii]

    u1  = ud[ii,:,:] 
    u2  = vd[ii,:,:] 
    b1 = b1d[ii,:,:]
    b2 = b2d[ii,:,:]

    q_hat = fftn(Q[ii,:,:])
    j_hat = fftn(j[ii,:,:])
    A_hat = fftn(A[ii,:,:])

    np.seterr(divide='ignore', invalid='ignore')
    psi_hat = -K2F2inv*q_hat

    gradq = zeros((2,N[1],N[0]))
    gradj = zeros((2,N[1],N[0]))
    gradA = zeros((2,N[1],N[0]))

    for i in range(2):
        gradq[i] = (ifftn( iK[i]*q_hat)).real
        gradj[i] = (ifftn( iK[i]*j_hat)).real  
        gradA[i] = (ifftn( iK[i]*A_hat)).real    

    KE_advect  = +(psi_hat*np.conjugate(fftn(u1*gradq[0]+u2*gradq[1]))).real 
    KE_lorentz = -(psi_hat*np.conjugate(fftn(b1*gradj[0]+b2*gradj[1]))).real
    KE_viscous = -2*(q_hat*np.conjugate(q_hat)/Re).real  
    ME_advect  = +(j_hat*np.conjugate(fftn(u1*gradA[0]+u2*gradA[1]))).real
    ME_viscous = -2*(j_hat*np.conjugate(j_hat)/Rm).real  

    KE_advect_bin  =  np.zeros(kbin.shape)
    KE_lorentz_bin =  np.zeros(kbin.shape)
    KE_viscous_bin =  np.zeros(kbin.shape) 
    ME_advect_bin  =  np.zeros(kbin.shape)
    ME_viscous_bin =  np.zeros(kbin.shape)

    for jj in range(0, len(kbin)-1):
        mask = np.logical_and(RP > kbin[jj], RP < kbin[jj+1]).astype(int)
        area = np.pi*(kbin[jj+1]**2-kbin[jj]**2)
        KE_advect_bin[jj] = ndimage.mean(np.fft.fftshift(KE_advect).real, mask)*area
        KE_lorentz_bin[jj] = ndimage.mean(np.fft.fftshift(KE_lorentz).real, mask)*area
        KE_viscous_bin[jj] = ndimage.mean(np.fft.fftshift(KE_viscous).real, mask)*area
        ME_advect_bin[jj] = ndimage.mean(np.fft.fftshift(ME_advect).real, mask)*area
        ME_viscous_bin[jj] = ndimage.mean(np.fft.fftshift(ME_viscous).real, mask)*area

    plt.clf()  # Plot rad integrated  

    KEadv_vec[ii,:] = KE_advect_bin 
    KElor_vec[ii,:] = KE_lorentz_bin
    MEadv_vec[ii,:] = ME_advect_bin
    KEvis_vec[ii,:] = KE_viscous_bin
    MEvis_vec[ii,:] = ME_viscous_bin
    
    plt.subplot(1,2,1)
    plt.xscale('log')
    plt.plot(kbin[1:], KE_advect_bin[1:], '-g', label=r'$T_{q}(k)$')
    plt.plot(kbin[1:], KE_lorentz_bin[1:], '-b', label=r'$T_{L}(k)$')
    plt.plot(kbin[1:], ME_advect_bin[1:], '-r', label=r'$T_{A}(k)$')
    plt.plot(kbin[1:],KE_advect_bin[1:]+KE_lorentz_bin[1:]+ ME_advect_bin[1:], '-k', label=r'T(k)')

    #max_scale = max()
    #yl = max(-np.max(np.abs(Tk[1:])) , np.max(np.abs(Tk[1:]))) 
    #plt.ylim([-yl,yl])
    
    plt.grid(True)
    plt.legend(loc='upper right');
    plt.xlabel("wavenumber (k)")
    plt.title('Dynamic Transfers at t = %4.2f' % t)

    plt.subplot(1,2,2)
    plt.xscale('log')
    plt.plot(kbin[1:], KE_viscous_bin[1:], '-r', label=r'$-2\mu k^2\hat E_V$')
    plt.plot(kbin[1:], ME_viscous_bin[1:], '-b', label=r'$-2\eta k^2\hat E_M$')
    plt.grid(True)
    plt.legend(loc='upper left');
    plt.xlabel("wavenumber (k)")
    plt.title('Viscous Transfers at t = %4.2f' % t)


    plt.draw()
    plt.pause(0.0001)
    plt.tight_layout()
    #if movie:
    #    plt.savefig('frame_{0:04d}.png'.format(ii), dpi=200)

if movie:
    merge_to_mp4('frame_%04d.png', movie_name)

plt.figure(figsize=(9,4))

# temporal averaging

t_begin = 5; first = int(np.where(np.array(tp)==t_begin)[0])
t_end   = 10; last = int(np.where(np.array(tp)==t_end)[0])

KEvis = np.mean(KEvis_vec[first:last,:],axis=0)
MEvis = np.mean(MEvis_vec[first:last,:],axis=0)
KEadv = np.mean(KEadv_vec[first:last,:],axis=0)
KElor = np.mean(KElor_vec[first:last,:],axis=0)
MEadv = np.mean(MEadv_vec[first:last,:],axis=0)
Total = np.mean(KEadv_vec[first:last,:]+KElor_vec[first:last,:]+MEadv_vec[first:last,:],axis=0)

plt.subplot(1,2,1)
#plt.xscale('log')
plt.plot(kbin[1:], KEadv[1:], '-g', label=r'$T_{q}(k)$')
plt.plot(kbin[1:], KElor[1:], '-b', label=r'$T_{L}(k)$')
plt.plot(kbin[1:], MEadv[1:], '-r', label=r'$T_{A}(k)$')
plt.plot(kbin[1:], Total[1:], '-k', label=r'$T(k)$')
#plt.xlim([0,180])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid(True)
plt.legend(loc='best');
plt.xlabel("wavenumber (k)")
plt.title('Time-averaged Dynamic Transfers')

plt.subplot(1,2,2)
#plt.xscale('log')
plt.plot(kbin[1:], KEvis[1:], '-r', label=r'$-2\mu k^2\hat E_V$')
plt.plot(kbin[1:], MEvis[1:], '-b', label=r'$-2\eta k^2\hat E_M$')
plt.grid(True)
#plt.xlim([0,180])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend(loc='best');
plt.xlabel("wavenumber (k)")
plt.title('Time-averaged Viscous Transfers')
plt.tight_layout()
plt.show()

file_name_png = "qgmhd_transfer_N{}.png".format(N[0])
#print('Saving plot in ', file_name_png)
#plt.savefig(file_name_png)

