#!/usr/bin/env python

# post-processing script to read the diagnostics output from the
# qgmhd_shenfun.py driver script.
#
# Current outputs: energy, anisotropy, lorentz force, microscales
# More Options: growth rates, conserved quantities (to uncomment)

# imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

# set resolution to pick our correct files
N = [1024]*2

# file extraction for diagnostics (after running generate_diagnostics.py)
folder = 'output-qgmhd'
file_name = folder + "/qgmhd_Nx{}_realdiagnostics.h5".format(N[0])
print('Read from file', file_name)

# read domain variables
file2  = h5py.File(file_name, "r")
domain = file2['domain']
assert(N[0] == file2['size'][0]), "Number of X domain points doesn't match!"
assert(N[1] == file2['size'][1]), "Number of Y domain points doesn't match!"

# read physical parameters
F2     = file2['F2'][...]
M2     = file2['M2'][...]
Uj     = file2['Uj'][...]
B0     = file2['B0'][...]
MHD    = file2['MHD'][...]
Re     = file2['Re'][...]
Rm     = file2['Rm'][...]

# read temporal parameters
times  = file2['times']
twrite = file2['twrite']
t0,tf,dt = times[0], times[1], times[2]

# create appropriate time vector
tt     = np.arange(t0,tf,dt)

# Display extracted physical parameters
print(' ')
print('Parameters')
print('==========')
print('F2 = ', F2)
print('M2 = ', M2)
print('Uj = ', Uj)
print('B0 = ', B0)
print('N  = (%d,%d) ' % (N[0],N[1]))
print(' ')

# pull diagnostics from file
vals   = file2['values']

KE    = vals[:,0]   # kinetic energy
PE    = vals[:,1]   # porential energy
Q2    = vals[:,2]   # <q^2>
Q1    = vals[:,3]   # <q>
qgrow = vals[:,4]   # ||q_pert||
ME    = vals[:,5]   # magnetic energy
A2    = vals[:,6]   # <A^2>
A1    = vals[:,7]   # <A>
j2    = vals[:,8]   # <j^2>
Agrow = vals[:,9]   # ||A_pert||
Hel   = vals[:,10]  # <u.b>
u12    = vals[:,11] # <u^2>
u22    = vals[:,12] # <v^2>
b12    = vals[:,13] # <b1^2>
b22    = vals[:,14] # <b2^2>
qx2    = vals[:,15] # <q_x^2>
qy2    = vals[:,16] # <q_y^2>
jx2    = vals[:,17] # <j_x^2>
jy2    = vals[:,18] # <j_y^2>
lor    = vals[:,19] # <sqrt(b.grad(j))>
TE     = KE+PE+ME   # total energy

# Compute microscale Lengths
Lu = np.sqrt(u12+u22)/np.sqrt(Q2)
Lb = np.sqrt(b12+b22)/np.sqrt(j2)

# create figure
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(7, 8))

### Energy

axes[0,0].set_title('Energy')
axes[0,0].plot(tt, TE[:-1], lw=3, color='k', label='Total')
axes[0,0].plot(tt, KE[:-1], lw=3, color='r', label='Kinetic', dashes=[6,2])
axes[0,0].plot(tt, PE[:-1], lw=3, color='g', label='Potential', dashes=[6,2])
axes[0,0].plot(tt, ME[:-1], lw=3, color='b', label='Magnetic', dashes=[6,2])
axes[0,0].legend(loc='best')
axes[0,0].grid(True)
axes[0,0].set_xlabel("t")

### Anisotropy measure

axes[0,1].set_title('Anisotropy Norm')
axes[0,1].plot(tt, u12[:-1]/(u12[:-1]+u22[:-1]), lw=3, color='r', label=r'$\frac{u_1^2}{u_1^2+u_2^2}$')
axes[0,1].plot(tt, b12[:-1]/(b12[:-1]+b22[:-1]), lw=3, color='b', label=r'$\frac{b_1^2}{b_1^2+b_2^2}$')
axes[0,1].plot(tt, qx2[:-1]/(qx2[:-1]+qy2[:-1]), lw=3, color='c', label=r'$\frac{q_x^2}{q_x^2+q_y^2}$')
axes[0,1].plot(tt, jx2[:-1]/(jx2[:-1]+jy2[:-1]), lw=3, color='m', label=r'$\frac{j_x^2}{j_x^2+j_y^2}$')
axes[0,1].legend(loc='best')
axes[0,1].grid(True)
axes[0,1].set_xlabel("t")

### Lorentz force

axes[1,0].set_title('Lorentz Force Norm')
axes[1,0].plot(tt, lor[:-1], lw=3, color='b', label=r'$M^2 \sqrt{(b \cdot \nabla j)^2}$')
axes[1,0].legend(loc='best')
axes[1,0].grid(True)
axes[1,0].set_xlabel("t")

### Conserved quantitites

#axes[1,0].set_title('Conserved quantities - initial')
#axes[1,0].plot(tt, (Hel[:-1]-Hel[0])/Hel[0], lw=3, color='c', label=r'$\frac{H-H(0)}{H(0)}$')
#axes[1,0].plot(tt, (A2[:-1]-A2[0])/A2[0],  lw=3, color='m', label=r'$\frac{A^2-A^2(0)}{A^2(0)}$')
#axes[1,0].legend(loc='best')
#axes[1,0].grid(True)
#axes[1,0].set_xlabel("t")

### Growth rate

#t_bound = [35, 40]
#I1 = np.array( np.where(((tt > t_bound[0]) & (tt < t_bound[1]))) )
#I1 = I1.ravel()

#if len(I1) > 2:
#    p1 = np.polyfit(tt[I1], np.log(qgrow[I1]), 1)
#    pp1 = np.exp(np.polyval(p1,(tt)))
#    p2 = np.polyfit(tt[I1], np.log(Agrow[I1]), 1)
#    pp2 = np.exp(np.polyval(p2,(tt)))
#    print('q and A growth rates are ', p1[0], p2[0])

#axes[1,0].set_title('Perturbation norms')
#axes[1,0].semilogy(tt, qgrow[:-1],  lw=3, color='r', label=r'$||q-\bar{q}||$')
#axes[1,0].semilogy(tt, Agrow[:-1],  lw=3, color='b', label=r'$||A-\bar{A}||$')
#if len(I1) > 2:
#    axes[1,0].plot(tt[I1], pp1[I1], '--m',  lw=3, label="q slope: {p:8.4f}".format(p=p1[0]))
#    axes[1,0].plot(tt[I1], pp2[I1], '--c',  lw=3, label="A slope: {p:8.4f}".format(p=p2[0]))
#axes[1,0].grid(True)
#axes[1,0].legend(loc='best')
#plt.xlim([tt[0], tt[-1]])
#axes[1,0].set_xlabel("t")

### L_u and L_b
axes[1,1].semilogy(tt, Lu[:-1], lw=3, color='r', label=r'$L_u$')
axes[1,1].semilogy(tt, Lb[:-1], lw=3, color='b', label=r'$L_b$')
axes[1,1].set_title('Microscales')
axes[1,1].legend(loc='best')
axes[1,1].grid(True)
axes[1,1].set_xlabel("t")
plt.xlim([tt[0], tt[-1]])

fig.suptitle('Diagnostics for QGMHD')
fig.tight_layout()

file_name_png = "qgmhd_gen_diag_N{}.png".format(N[0])
print('Saving plot in ', file_name_png)
plt.savefig(file_name_png)

