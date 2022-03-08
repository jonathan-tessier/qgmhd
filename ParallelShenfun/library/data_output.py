#!/usr/bin/env python

# src script used by qgmhd_shenfun.py to set up the data output. 
# Returns the H5 File structures and has a basic plotting function
# for plotting the PV and A on the fly (as sim is running).
# Further contains a diagnostic computation function for post-processing
# parallel runs.

# library imports
import numpy as np
from mpi4py import MPI
import netCDF4 as nc4
from shenfun import *
import h5py
import os
from datetime import datetime
import matplotlib.pyplot as plt
from library.operators import inner, cross
today = datetime.today()
import subprocess

# output stuff
def initialize_output(phys, times, grid, folder):

    path   = os.getcwd() + '/' + folder

    try: # Create directory
        os.mkdir(path)
    except OSError:
        print ("Directory for data files already exists: %s" % path)
    else:
        print ("Storying data files in the directory: %s " % path)

    file0name = folder + '/qgmhd_Nx%d_fields' % (grid.Nx) + '.h5'
    file0 = HDF5File(file0name, mode='w', domain=grid.domain)

    rank = MPI.COMM_WORLD.rank

    # h5 file for diagnostics
    fileh5 = folder + '/qgmhd_Nx%d_diagnostics' % (grid.Nx) + '.h5'
    fileh5py = h5py.File(fileh5, "w", driver='mpio', comm=MPI.COMM_WORLD)

    norms_vals   = fileh5py.create_dataset("values", (times.Nt,len(phys.diag_fields)), dtype='double')
    norms_times  = fileh5py.create_dataset("times",  (3,),  data=(times.t0, times.tf+times.dt, times.dt))
    norms_twrite = fileh5py.create_dataset("twrite", data=times.tplot)
    norms_domain = fileh5py.create_dataset("domain", data=grid.domain)
    norms_N      = fileh5py.create_dataset("size",   data=[grid.Ny, grid.Nx])
    norms_F2     = fileh5py.create_dataset("F2",     data=phys.F2)
    norms_M2     = fileh5py.create_dataset("M2",     data=phys.M2)
    norms_Uj     = fileh5py.create_dataset("Uj",     data=phys.Uj)
    norms_B0     = fileh5py.create_dataset("B0",     data=phys.B0)
    norms_Re     = fileh5py.create_dataset("Re",     data=phys.Re)
    norms_Rm     = fileh5py.create_dataset("Rm",     data=phys.Rm)
    norms_psi_amp= fileh5py.create_dataset("psi_amp",data=phys.psi_amp)
    norms_A_amp  = fileh5py.create_dataset("A_amp",  data=phys.A_amp)
    norms_MHD    = fileh5py.create_dataset("MHD",    data=phys.MHD)
    
    fileh5py.attrs['names'] = np.array(phys.diag_fields, dtype='S')
    return file0, norms_vals

# plotting on the fly
def plot_field(soln, t):

    if t == 0:
        plt.figure(figsize=(10,4))
    else:
        plt.clf()

    plt.subplot(1,2,1)
    plt.pcolormesh(soln.grid.X[0],soln.grid.X[1],soln.qA[0], cmap='seismic', shading='gouraud')
    plt.clim([-np.amax(soln.qA[0]),np.amax(soln.qA[0])])
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar()
    plt.title('q perturbation at t=%4.3f' %t)
    plt.subplot(1,2,2)
    plt.pcolormesh(soln.grid.X[0],soln.grid.X[1],soln.qA[1], cmap='seismic', shading='gouraud')
    plt.clim([-np.amax(soln.qA[1]),np.amax(soln.qA[1])])
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar() 
    plt.title('A perturbation at t=%4.3f' %t)

    plt.draw()
    plt.pause(0.001)

# for parallel runs, this creates the global diagnostics from the saved fields.
def generate_diagnostics(phys,times,grid,folder,T0):
    # set resolution to pick our correct files
    N = grid.N

    # file extraction for current diagnostics
    folder = folder
    domain = grid.domain

    # read physical parameters
    F2     = phys.F2
    M2     = phys.M2
    Uj     = phys.Uj
    B0     = phys.B0
    MHD    = phys.MHD
    Re     = phys.Re
    Rm     = phys.Rm
    psi_amp= phys.psi_amp
    A_amp  = phys.A_amp

    # read temporal parameters
    t0,tf,dt = (times.t0, times.tf+times.dt, times.dt) # contains to, tf, dt
    twrite = times.tplot  # contains plotting frequency

    # create appropriate time vector
    tt       = np.arange(t0, tf, dt)
    ntplot   = int(twrite/dt)
    tplot    = tt[0::ntplot]
    dtplot   = tplot[1]-tplot[0]
    Nt       = len(tplot)

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

    # dataset generation
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
    T   = T0
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

# movie function
def merge_to_mp4(frame_filenames, movie_name, fps=12):

    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    f_log.close()
    f_err.close()

# function for tenth power formatting for M2 variable
def fmt(x):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if float(a) == 0: 
        return "0"
    elif float(a) == 1:
        if b == 0:
            return "1"
        else:
            return r'$10^{{{}}}$'.format(b)
    else:
        return r'${} \times 10^{{{}}}$'.format(a, b)

# function to return the index of the axes object in the subplots
def panelindex(index,ncols,nrows):
    if nrows == 1 or ncols == 1:
        return (index,)
    elif nrows>1 and ncols>1:
        return np.divmod(index,ncols)
    elif nrows==1 and ncols==1:
        raise(IndexError("This code can't plot a single simulation. Use the basic scripts instead."))
    else:
        raise(IndexError("Figure Panel Configuration doesn't work."))

