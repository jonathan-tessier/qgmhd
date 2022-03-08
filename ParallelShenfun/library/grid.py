#!/usr/bin/env python

# src script to define the spectral quantities used by the qg_physics.py and time_stepping.py
# to evolve the PV and Q equations. Further defines the spectral filter and random function 
# generators to study turbulence.

import numpy as np
from shenfun import *

# Set up params stuff        
class grid_parameters(object):

    def __init__(
            self,
            phys,
            T,
            Nx = 128,
            Ny = 128,
            Lx = 8*np.pi,
            Ly = 8*np.pi,
            display = True,
            filt = "smooth"
    ):

        """
        Grid Parameters
        """

        self.phys = phys
        self.filt = filt

        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx/self.Nx
        self.dy = Ly/self.Ny

        self.L  = (Lx, Ly)
        self.dim = len(self.L)
        self.N  = (self.Nx, self.Ny)
        self.domain = ((0., self.Lx), (0., self.Ly))
        self.T  = T

        self.X     = self.T.local_mesh(True)
        #x = np.linspace(self.dx/2, Lx-self.dx/2, self.Nx)
        #y = np.linspace(self.dy/2, Ly-self.dy/2, self.Ny)      
        #X = np.array( np.meshgrid(x, y) )

        self.K     = np.array(self.T.local_wavenumbers(True,True))
        #self.kx = 2*np.pi/Lx*np.hstack([range(0,int(self.Nx/2)+1), range(-int(self.Nx/2)+1,0)])
        #self.ky = 2*np.pi/Ly*np.hstack([range(0,int(self.Ny/2)+1), range(-int(self.Ny/2)+1,0)])
        #K = np.array( np.meshgrid(kx, ky, indexing='ij') )

        np.seterr(divide='ignore', invalid='ignore')
        self.iK      = 1j*self.K
        self.K2      = np.sum(self.K*self.K, 0, dtype=float)
        self.K2F2    = np.sum(self.K*self.K + phys.F2, 0, dtype=float)
        self.K2F2inv = np.where(self.K2F2 == 0.0, 0.0, 1.0/self.K2F2).astype(float)
        self.iKoK2F2 = 1j*self.K.astype(float) * self.K2F2inv
        
        # Filter Parameters
        kmax = np.amax(abs(self.K));
        if filt=="smooth":
            # Smooth Gaussian Filter (Works at any resolution)
            ks = 0.8*kmax;
            km = 0.9*kmax;
            alpha = 0.69*ks**(-1.88/np.log(km/ks));
            beta  = 1.88/np.log(km/ks);
            self.sfilt = np.exp(-alpha*(self.K[0]**2 + self.K[1]**2)**(beta/2.0));
        elif filt=="sharp":
            # option for a 2/3 filter (Sharp) Needs high resolution to be non-divergent
            kfilter = kmax*2/3
            self.sfilt = (np.sqrt(self.K[0]**2 + self.K[1]**2)<=kfilter).astype(float)
        else:
            raise(NameError("Filter Definition not recognized. Must be smooth or sharp"))

        self.display = display
        if self.display:
            print(' ')
            print('Grid Parameters')
            print('===============')
            print('Nx  = ', self.Nx, '\nNy  = ', self.Ny, \
                  '\nLx  = ', self.Lx, '\nLy  = ', self.Ly)
            print(' ')


# random field generation: Annulus in spectral space about kstar with spectral width sigma
def generate_field_with_kpeak(T,size,grid,kstar,sigma,npseed,amp):

    kxx, kyy = grid.K[0], grid.K[1]
    krr = np.sqrt(kxx**2  + kyy**2)
    kmin, kmax  = kstar - sigma/2, kstar + sigma/2
    window = (kmin <= krr) * (kmax >= krr)
    noise = Function(T); np.random.seed(npseed)
    noise = T.forward(np.random.uniform(-1., 1., size=size), noise)

    fhat = np.multiply(noise, window.astype(float))
    f = Array(T)
    f = T.backward(fhat,f)
    f = f.real
    f = f-np.mean(f)
    f = amp*f/np.max(abs(f))

    return f

# random field generation: Gaussian bump in spectral space (see window definition)
def generate_broad_spectrum_field(T,size,grid,kstar,sigma,npseed,amp):

    kxx, kyy = grid.K[0], grid.K[1]
    krr = np.sqrt(kxx**2  + kyy**2)
    window = np.exp(-(krr/(sigma*kstar))**2)
    noise = Function(T); np.random.seed(npseed)
    noise = T.forward(np.random.uniform(-1., 1., size=size), noise)

    fhat = np.multiply(noise, window.astype(float))
    f = Array(T)
    f = T.backward(fhat,f)
    f = f.real
    f = f-np.mean(f)
    f = amp*f/np.max(abs(f))

    return f
