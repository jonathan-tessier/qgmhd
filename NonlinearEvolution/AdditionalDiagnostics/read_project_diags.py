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
#qgrow = vals[:,4]

ME    = vals[:,5]
A2    = vals[:,6]
A1    = vals[:,7]
j1    = vals[:,8]
#Agrow = vals[:,9]
Hel   = vals[:,10]

TE    = KE + PE + ME

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(8, 10))

### Energy

axes[0,0].set_title('Energy')
axes[0,0].plot(tt, KE[:-1], lw=3, color='b', label=r'$E_V$')
#axes[0,0].plot(tt, PE[:-1], lw=3, color='g', label='Potential')
axes[0,0].plot(tt, ME[:-1], lw=3, color='r', label=r'$E_M$')
axes[0,0].legend(loc='best')
axes[0,0].grid(True)

print('Means: KE = %6.6e, PE = %6.6e, ME = %6.6e' % \
      ( np.mean(KE[2:-1]), np.mean(PE[2:-2]), np.mean(ME[2:-2])))

### Conserved quantitites

axes[0,1].set_title('Conserved scalars')
axes[0,1].plot(tt, (Hel[:-1]/abs(Hel[0])), lw=3, color='b', label=r'$H/H(0)$')
axes[0,1].plot(tt, (A2[:-1]/A2[0]),  lw=3, color='r', label=r'$A^2/A^2(0)$')
axes[0,1].legend(loc='best')
axes[0,1].grid(True)

axes[1,0].set_title('Energy Ratio')
axes[1,0].plot(tt, KE[:-1]/ME[:-1], lw=3, color='k', label=r'$E_V/E_M$')
axes[1,0].legend(loc='best')
axes[1,0].grid(True)

axes[1,1].set_title('Total Energy')
axes[1,1].plot(tt, TE[:-1]/TE[0], lw=3, color='k', label='E/E(0)')
axes[1,1].legend(loc='best')
axes[1,1].grid(True)

fig.suptitle('Diagnostics for QGMHD (pyfftw)')
fig.tight_layout()
plt.show()

file_name_png = "qgmhd_diag_N{}.png".format(N[0])
#print('Saving plot in ', file_name_png)
#plt.savefig(file_name_png)


