#!/usr/bin/env python

# For parallel runs of qgmhd_shenfun.py, copy this script to the same
# directory containing output-qgmhd and run to combine diagnostics from
# all core to the global variables. $ python3 generate_diagnostics.py

# Deprecated: Updated driver script 'qgmhd_shenfun.py' runs this code
# automatically after simulation is complete. Use this script to recompute
# diagnostics or to deal with previous runs where parallel diags were not
# computed automatically

# imports
import numpy as np
import h5py
from mpi4py import MPI
from shenfun import *
from library.operators import inner, cross

# set resolution to pick our correct files
N = [1024]*2

# file extraction for current diagnostics
folder = 'output-qgmhd'
file_name = folder + "/qgmhd_Nx{}_diagnostics.h5".format(N[0])
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
psi_amp= file2['psi_amp'][...]
A_amp  = file2['A_amp'][...]

# read temporal parameters
times  = file2['times']  # contains to, tf, dt
twrite = file2['twrite'] # contains plotting frequency
t0,tf,dt = times[0], times[1], times[2]

# create appropriate time vector
tt       = np.arange(t0, tf, dt)
ntplot   = int(twrite[...]/times[2])
tplot    = tt[0::ntplot]
dtplot   = tplot[1]-tplot[0]
Nt       = len(tplot)

# Display extracted physical parameters
print('Parameters')
print('==========')
print('F2 = ', F2)
print('M2 = ', M2)
print('Uj = ', Uj)
print('B0 = ', B0)
print('N  = (%d,%d) ' % (N[0],N[1]))
print(' ')

# file extraction for computed fields
file_name = folder + "/qgmhd_Nx{}_fields.h5".format(N[0])
print('Read from file', file_name)

# Get information from file
f = h5py.File(file_name, 'r')            # open file
keys = list(f.keys())                    # list variables
var1 = keys[0]                           # pick the first field

# domain variables
x0 = np.array(f[var1 + '/domain/x0'][()])       # bounds in 0-axis
x1 = np.array(f[var1 + '/domain/x1'][()])       # bounds in 1-axis

# print domain lengths
print('Lengths of domain:')
print('      x0 in [%6.4f, %6.4f]' % (x0[0], x0[-1]))
print('      x1 in [%6.4f, %6.4f]' % (x1[0], x1[-1]))

# save domain lengths
L = (x0[-1], x1[-1])
# compute the domain area
Area_xy = L[0]*L[1]

# pull background fields from file
q_bar = np.array(f['q_bar/2D/0'][()])
A_bar = np.array(f['A_bar/2D/0'][()])
u_bar = np.array(f['u_bar/2D/0'][()])
b_bar = np.array(f['b_bar/2D/0'][()])
psi_bar = np.array(f['psi_bar/2D/0'][()])

# file to write global diagnostics
file_name = folder + "/qgmhd_Nx{}_realdiagnostics.h5".format(N[0])
print('Going to write to file', file_name)

# dataset geenration
fileh5py     = h5py.File(file_name, "w")
norms_vals   = fileh5py.create_dataset("values", (Nt,20), dtype='double')
norms_times  = fileh5py.create_dataset("times",  (3,),  data=(t0, tf-dtplot, dtplot))
norms_twrite = fileh5py.create_dataset("twrite", data=tplot)
norms_domain = fileh5py.create_dataset("domain", data=domain)
norms_N      = fileh5py.create_dataset("size",   data=[N[1], N[0]])
norms_F2     = fileh5py.create_dataset("F2",     data=F2)
norms_M2     = fileh5py.create_dataset("M2",     data=M2)
norms_Uj     = fileh5py.create_dataset("Uj",     data=Uj)
norms_B0     = fileh5py.create_dataset("B0",     data=B0)
norms_Re     = fileh5py.create_dataset("Re",     data=Re)
norms_Rm     = fileh5py.create_dataset("Rm",     data=Rm)
norms_psi_amp= fileh5py.create_dataset("psi_amp",data=psi_amp)
norms_A_amp  = fileh5py.create_dataset("A_amp",  data=A_amp)
norms_MHD    = fileh5py.create_dataset("MHD",    data=MHD)

# norm functions for the diagnostics
def mean_norm(field):
    return np.mean(field)*Area_xy

def l2_norm(field):
    return np.linalg.norm(field)/N[0]

# Create necessary function spaces and spectral variables
V1  = FunctionSpace(N[0], 'F', dtype='D', domain=(0, L[0]))
V2  = FunctionSpace(N[1], 'F', dtype='d', domain=(0, L[1]))
T   = TensorProductSpace(MPI.COMM_SELF, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
TV  = VectorSpace(T)
X   = T.local_mesh(True)
K   = T.local_wavenumbers(True,True)

# init gradients
gradq = Array(TV)
gradj = Array(TV)

# keeps magnetic energy zero but will plot the other diagnostics for the hydro case.
if M2 == 0: M2 = 1;

# main loop to recreate diagnostics from output fields
for ii in np.arange(0, Nt):

    # read perturbation fields
    q = np.array(f['q/2D/' + str(ii)][()])  
    j = np.array(f['j/2D/' + str(ii)][()])  
    u = np.array(f['u/2D/' + str(ii)][()])
    b = np.array(f['b/2D/' + str(ii)][()])
    A = np.array(f['A/2D/' + str(ii)][()])
    psi = np.array(f['psi/2D/' + str(ii)][()])

    # create total fields
    q_total = q + q_bar
    A_total = A + A_bar
    u_total = u + u_bar
    b_total = b + b_bar
    psi_total = psi + psi_bar

    # compute gradients
    for i in range(2):
        gradq[i][:] = T.backward(1j*K[i]*T.forward(q_total))
        gradj[i][:] = T.backward(1j*K[i]*T.forward(j))

    # compute diagnostics
    norms_vals[ii]  = np.array((
        mean_norm(0.5*inner(u_total,u_total)),        # KE
        mean_norm(0.5*F2*psi_total**2),               # PE
        mean_norm(q_total**2),                        # q2
        mean_norm(q_total),                           # q
        l2_norm(q),                                   # q_pert
        mean_norm(0.5*MHD*M2*inner(b_total,b_total)), # ME
        mean_norm(M2*A_total**2),                     # A2
        mean_norm(np.sqrt(M2)*A_total),               # A
        mean_norm(M2*j**2),                           # j2
        l2_norm(np.sqrt(M2)*A),                       # A_pert
        mean_norm(inner(u_total,np.sqrt(M2)*b_total)),# H..
        mean_norm(u_total[0]**2),                     # u^2 for anisotropy measure
        mean_norm(u_total[1]**2),                     # v^2 for anisotropy measure
        mean_norm(M2*b_total[0]**2),                  # b1^2 for anisotropy measure
        mean_norm(M2*b_total[1]**2),                  # b2^2 for anisotropy measure
        mean_norm(gradq[0]**2),                       # q_x^2 for anisotropy measure
        mean_norm(gradq[1]**2),                       # q_y^2 for anisotropy measure
        mean_norm(M2*gradj[0]**2),                    # j_x^2 for anisotropy measure
        mean_norm(M2*gradj[1]**2),                    # j_y^2 for anisotropy measure
        mean_norm(M2*np.sqrt(inner(b_total,gradj)**2))# lorentz force bulk quantity
    ))
print("Done.")


