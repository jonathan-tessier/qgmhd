import numpy as np

# Basic inner product of vectors
def inner(u, v):
    return u[0]*v[0] + u[1]*v[1]

# Basic Jacobian of vectors
def jacobian(u, v):
    return u[0]*v[1] - u[1]*v[0]
    

def jet_bickley(phys, grid, soln):
        
    soln.q        =   2.*phys.Uj*np.tanh(grid.yy - grid.Ly/2.)/pow(np.cosh(grid.yy - grid.Ly/2.),2) + phys.F2*phys.Uj*np.tanh(grid.yy - grid.Ly/2.)
    soln.gradq[1] =   4.*phys.Uj*(0.5 - (np.sinh(grid.yy - grid.Ly/2)**2))/(np.cosh(grid.yy - grid.Ly/2))**4 + phys.F2*phys.Uj/pow(np.cosh(grid.yy - grid.Ly/2.),2)
    soln.u[0]     =   phys.Uj/pow(np.cosh(grid.yy - grid.Ly/2),2)
    soln.A        = - grid.yy
    soln.b[0]     =   1. + 0*grid.xx
    #soln.j = 0; #soln.gradj[i] = 0

def vortex_orszag_tang(phys, grid, soln):
    
    # see: Biskamp and Welter 1989 
    vxphase = 1.4; vyphase = 0.5; axphase = 2.3; ayphase = 4.1; ascale = 1/3; 

    soln.q        = - (np.cos(grid.xx+vxphase) + np.cos(grid.yy+vyphase))
    soln.gradq[0] =   np.sin(grid.xx+vxphase)
    soln.gradq[1] =   np.sin(grid.yy+vyphase)
    soln.u[0]     =   np.sin(grid.yy+vyphase)
    soln.u[1]     = - np.sin(grid.xx+vxphase)
    soln.A        =   ascale*(np.cos(2*grid.xx+axphase) + np.cos(grid.yy+ayphase))
    soln.b[0]     =   ascale*np.sin(grid.yy+ayphase)
    soln.b[1]     = - ascale*2*np.sin(2*grid.xx+axphase)
    soln.j        = - ascale*(4*np.cos(2*grid.xx+axphase) + np.cos(grid.yy+ayphase))
    soln.gradj[0] =   ascale*8*np.sin(2*grid.xx+axphase)
    soln.gradj[1] =   ascale*np.sin(grid.yy+ayphase)
