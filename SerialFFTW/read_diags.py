import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

M = 7
N = (2**M, 2**M)

folder           = 'output-qgmhd/'
file_name = folder + "qgmhd_Nx{}_diagnostics.h5".format(N[0])
print('Read from file', file_name)
file2 = h5py.File(file_name, "r")

vals  = file2['values']
times = file2['times']
tt    = np.arange(times[0], times[1], times[2])
#print(file2.attrs['names'][0])
#print(file2.attrs['names'][1])
#print(file2.attrs['names'][2])

M2 = file2['M2'][0]
F2 = file2['F2'][0]
Uj = file2['Uj'][0]
Re = file2['Re'][0]
Rm = file2['Rm'][0]
amp = file2['amp'][0]

print(' ')
print('Physical Parameters')
print('===================')
print('F2    = ', F2, '\nM2    = ', M2, \
      '\nRm    = ', Rm, '\nRe    = ', Re, \
      '\nUj    = ', Uj, '\namp   = ', amp)
print(' ')

KE    = vals[:,0]
PE    = vals[:,1]
Q2    = vals[:,2]
Q1    = vals[:,3]
qgrow = vals[:,4]

ME    = vals[:,5]
A2    = vals[:,6]
A1    = vals[:,7]
j1    = vals[:,8]
Agrow = vals[:,9]
Hel   = vals[:,10]

TE    = KE + PE + ME

# Microscale Lengths
Lu = 2*KE/Q1
Lb = 2*ME/j1

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(8, 10))

### Energy

axes[0,0].set_title('Energy')
axes[0,0].plot(tt, KE[:-1], lw=3, color='b', label='Kinetic')
axes[0,0].plot(tt, PE[:-1], lw=3, color='g', label='Potential')
axes[0,0].plot(tt, ME[:-1], lw=3, color='m', label='Magnetic', dashes=[6,2])
axes[0,0].plot(tt, TE[:-1], lw=3, color='r', label='Total', dashes=[6,2])
axes[0,0].legend(loc='best')
axes[0,0].grid(True)

print('Means: KE = %6.6e, PE = %6.6e, ME = %6.6e' % \
      ( np.mean(KE[2:-1]), np.mean(PE[2:-2]), np.mean(ME[2:-2])))

### Conserved quantitites

axes[0,1].set_title('Conserved quantities - Initial')
axes[0,1].plot(tt, (Hel[:-1] - Hel[0]), lw=3, color='y', label='Helicity')
axes[0,1].plot(tt, (A2[:-1] - A2[0]),  lw=3, color='m', label='A2')
axes[0,1].legend(loc='best')
axes[0,1].grid(True)

### Growth rate

t_bound = [50, 60]
I1 = np.array( np.where(((tt > t_bound[0]) & (tt < t_bound[1]))) )
I1 = I1.ravel()

if len(I1) > 2:
    p1 = np.polyfit(tt[I1], np.log(qgrow[I1]), 1)
    pp1 = 4e-7*np.exp(p1[0]*tt[I1])
    p2 = np.polyfit(tt[I1], np.log(Agrow[I1]), 1)
    pp2 = 1e-5*np.exp(p2[0]*tt[I1])

    print('q and A growth rates are ', p1[0], p2[0])

axes[1,0].set_title('Growth rates')
axes[1,0].semilogy(tt, qgrow[:-1],  lw=3, color='b', label='pert q')
axes[1,0].semilogy(tt, Agrow[:-1],  lw=3, color='m', label='pert A')
if len(I1) > 2:
    axes[1,0].plot(tt[I1],      pp1, '--r',  lw=3, label='fit q')
    axes[1,0].plot(tt[I1],      pp2, '--g',  lw=3, label='fit A')
axes[1,0].grid(True)
axes[1,0].legend(loc='best')
plt.xlim([tt[0], tt[-1]])

### L_u and L_b
axes[1,1].semilogy(tt, Lu[:-1], lw=3, color='b', label='Lu')
axes[1,1].semilogy(tt, Lb[:-1], lw=3, color='m', label='Lb')
axes[1,1].set_title('Length scales')
axes[1,1].legend(loc='best')
axes[1,1].grid(True)
plt.xlim([tt[0], tt[-1]])

print('Means: Lu = %6.6e, Lb = %6.6e' % ( np.mean(Lu[2:-2]), np.mean(Lb[2:-2])))

fig.suptitle('Diagnostics for QGMHD (pyfftw)')
fig.tight_layout()
plt.show()

file_name_png = "qgmhd_diag_N{}.png".format(N[0])
#print('Saving plot in ', file_name_png)
#plt.savefig(file_name_png)


