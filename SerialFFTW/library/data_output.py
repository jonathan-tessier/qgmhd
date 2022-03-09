import numpy as np
import netCDF4 as nc4
import h5py
import os

from datetime import datetime

today = datetime.today()


#######################################################
#        Create netCDF info                           #
#######################################################

# output stuff
def build_netcdf(rootgrp, times, grid):


    #FJP: update
    # Open netcdf4 file
    # rootgrp
    #        -> solngrp
    #                     -> qgrp : potential vorticity
    #                     -> Agrp : magnetic streamfunction
   
    # move into function
    rootgrp.description = "Data for one-layer QG MHD model"
    rootgrp.history = "Created " + today.strftime("%d/%m/%y")

    # Create group
    solngrp    = rootgrp.createGroup('soln')

    # Specify dimensions
    Nt       = rootgrp.createDimension('time',      times.Nt)
    timeplot = rootgrp.createDimension('timeplot',  len(times.ttplt))
    Nx       = rootgrp.createDimension('x-dim',     grid.Nx)
    Ny       = rootgrp.createDimension('y-dim',     grid.Ny)

    # Build variables
    time     = rootgrp.createVariable('Time', 'f4','time')
    timeplot = rootgrp.createVariable('TimePlot', 'f4','timeplot')
    xvar     = rootgrp.createVariable('x', 'f4','x-dim')
    yvar     = rootgrp.createVariable('y', 'f4','y-dim')
    q        = rootgrp.createVariable('PV', 'd', ('timeplot','x-dim','y-dim') )
    A        = rootgrp.createVariable('A',  'd', ('timeplot','x-dim','y-dim') )
    j        = rootgrp.createVariable('j',  'd', ('timeplot','x-dim','y-dim') )
    u        = rootgrp.createVariable('u',  'd', ('timeplot','x-dim','y-dim') )
    v        = rootgrp.createVariable('v',  'd', ('timeplot','x-dim','y-dim') )
    b1       = rootgrp.createVariable('b1', 'd', ('timeplot','x-dim','y-dim') )
    b2       = rootgrp.createVariable('b2', 'd', ('timeplot','x-dim','y-dim') )
    
    time.units     = 's'
    timeplot.units = 's'
    xvar.units     = 'm'
    yvar.units     = 'm'
    q.units        = '1/s'
    A.units        = '?/s'
    j.units        = '?/s/m^2'
    u.units        = 'm/s'
    v.units        = 'm/s'
    b1.units       = 'm/s'
    b2.units       = 'm/s'

    time[:]     = times.tt
    timeplot[:] = times.ttplt
    xvar[:]     = grid.x
    yvar[:]     = grid.y
    
    return q, A, j, u, v, b1, b2

# output stuff
def initialize_output(phys, times, grid, folder):

    path   = os.getcwd() + '/' + folder

    try:                                                                  # Create directory
        os.mkdir(path)
    except OSError:
        print ("Directory for data files already exists: %s" % path)
    else:
        print ("Storying data files in the directory: %s " % path)

    # NetCDF file for variables
    filenc   = folder + '/qgmhd_Nx%d_variables' % (grid.Nx) + '.nc'
    rootgrp = nc4.Dataset(filenc,'w', format='NETCDF4')
    gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2 = build_netcdf(rootgrp, times, grid) 
    
    # h5 file for diagnostics
    fileh5 = folder + '/qgmhd_Nx%d_diagnostics' % (grid.Nx) + '.h5'
    fileh5py = h5py.File(fileh5, "w")
    norms_vals   = fileh5py.create_dataset("values", (times.Nt,len(phys.diag_fields)), dtype='f')
    norms_times  = fileh5py.create_dataset("times",  (3,),  data=(times.t0, times.tf+times.dt, times.dt))
    norms_domain = fileh5py.create_dataset("domain", data=grid.domain)
    norms_N      = fileh5py.create_dataset("size",   data=[grid.Ny, grid.Nx])
    norms_F2     = fileh5py.create_dataset("F2",     data=[phys.F2])
    norms_M2     = fileh5py.create_dataset("M2",     data=[phys.M2])
    norms_Uj     = fileh5py.create_dataset("Uj",     data=[phys.Uj])
    norms_Re     = fileh5py.create_dataset("Re",     data=[phys.Re])
    norms_Rm     = fileh5py.create_dataset("Rm",     data=[phys.Rm])
    norms_amp    = fileh5py.create_dataset("amp",    data=[phys.amp])
    
    fileh5py.attrs['names'] = np.array(phys.diag_fields, dtype='S')

    return gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2, norms_vals

# plotting, output stuff?
def plot_field(soln, t):

    if t == 0:
        plt.figure() #figsize=(30,12))
    else:
        plt.clf()

    plt.subplot(1,2,1)
    plt.pcolormesh(soln.grid.x,soln.grid.y,soln.q, shading='gouraud')
    plt.colorbar()
    plt.title('q perturbation at t=%4.3f' %t)
    plt.subplot(1,2,2)
    plt.pcolormesh(soln.grid.x,soln.grid.y,soln.A, shading='gouraud')
    plt.colorbar() 
    plt.title('A perturbation at t=%4.3f' %t)

    plt.draw()
    plt.pause(0.0001)

