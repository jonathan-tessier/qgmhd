# LSAQGMHD: Parallel processing - Full Problem - With Diffusion - With Beta

# BY: Jonathan Tessier, June 2020

import numpy as np
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import sys

try: # Try using mpi
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
except:
    rank = 0
    num_procs = 1
print("Proc {0:d} of {1:d}".format(rank+1,num_procs))

# Domain parameters
L     = 8*np.pi                    # Length of domain
N     = 512                        # Number of grid points

# Physical Parameters
F     = 0.25                       # Froude number
M2    = 1e-4                       # Magnetic number
beta  = 0.0                        # nondim beta from coriolis
Re    = 1e4                        # Reynolds Number
Rm    = 1e4                        # Magnetic Reynolds Number

Reinv = 1/Re # Inverse Reynolds number to allow for inviscid case
Rminv = 1/Rm # Inverse Reynolds number to allow for inviscid case

# Jet parameters
Lj = 1.0                           # width of jet
Uj = 1.0                           # maximum velocity of jet

method = 'cheb'

if rank==0:
    print("Computational Parameters:")
    print("Domain Length (L) = "+str(L))
    print("Domain Num Points = "+str(N))
    print("Domain Resolution = "+str(L/N))
    print("")
    print("Physical Parameters")
    print("F    = "+str(F))
    print("M2   = "+str(M2))
    print("beta ="+str(beta))
    print("Re   = "+str(Re))
    print("Rm   = "+str(Rm))

def cheb(N):
    if N == 0:
        D = 0
        x = 1
    else:
        x = np.cos(np.pi*np.array(range(0,N+1))/N).reshape([N+1,1])
        c = np.ravel(np.vstack([2, np.ones([N-1,1]), 2])) \
            *(-1)**np.ravel(np.array(range(0,N+1)))
        c = c.reshape(c.shape[0],1)
        X = np.tile(x,(1,N+1))
        dX = X-(X.conj().transpose())
        D  = (c*(1/c).conj().transpose())/(dX+(np.eye(N+1)))   # off-diagonal entries
        D  = D - np.diag(np.sum(D,1))   # diagonal entries
    return D,x

# Differential Operators
Dy,y = cheb(N)
y   = (y[:,0]+1)*L/2
Dy  = Dy*(2/L)
Dyy = np.dot(Dy,Dy)

# Define Useful Operators
I  = np.identity(N-1)
O  = np.zeros([N-1,N-1])

# Define Basic State
P  = -Uj*np.tanh((y-L/2)/Lj)
U  = Uj/(np.cosh((y-L/2)/Lj))**2
Q  = -F*P + np.dot(Dyy,P)  #+ beta*y
Q2 = -2.*Uj*np.tanh((y-L/2)/Lj)/(np.cosh((y-L/2)/Lj))**2
Uyy = np.dot(Dyy,U)
B0 = np.ones(np.shape(y)) 
Byy= np.dot(Dyy,B0)

# Define range of wavenumbers 
dk = 5e-2
kk = np.arange(dk,2+dk,dk)
Nk = len(kk)

# Define storage vectors: DIM = [psi, a]x[num modes]x[num wavenumbers] = 2 x Ne x Nk
Ne = 4
grow = np.zeros((Ne,Nk))
freq = np.zeros((Ne,Nk))
modes = np.zeros((2,N+1,Ne,Nk),dtype=complex)

# Loop over wavenumbers
cnt = 0
for cnt in range(rank,len(kk),num_procs):

    k = kk[cnt]
    k2 = k**2
    nabla = Dyy[1:-1,1:-1] - k2*I
    
    # DIFFUSION !!!!!!! F or F2 HERE??? CMP with SIM
    B = np.vstack((np.hstack((nabla - F*I, O)),
                  np.hstack((O, I))))
    A = np.vstack((np.hstack((np.dot((np.diag(U[1:-1],0)+1j*Reinv/k*(nabla - F*I)),nabla) - np.diag(Uyy[1:-1],0)+beta*I, 
                             -M2*(np.dot(np.diag(B0[1:-1],0),nabla) - np.diag(Byy[1:-1],0)))),
                  np.hstack((-np.diag(B0[1:-1],0), 
                             np.diag(U[1:-1],0)+1j*Rminv/k*(nabla)))))    
    
    # NO DIFFUSION
    #B = np.vstack((np.hstack((Dyy[1:-1,1:-1] - (k2 + F)*I, O)),
    #              np.hstack((O, I))))
    #A = np.vstack((np.hstack((np.dot(np.diag(U[1:-1],0),Dyy[1:-1,1:-1] - (k2+F)*I) + np.diag(Qy[1:-1],0)+beta*I, 
    #                         -M**2*(np.dot(np.diag(B0[1:-1],0),Dyy[1:-1,1:-1] - (k2)*I) - np.diag(Byy[1:-1],0)))),
    #              np.hstack((-np.diag(B0[1:-1],0), 
    #                         np.diag(U[1:-1],0)))))

    # Solve for eigenvalues
    eigVals,eigVecs = spalg.eig(A,B)

    # Sort eigenvalues and eigenvectors
    ind = (-np.imag(eigVals)).argsort()
    eigVecs = eigVecs[:,ind]
    eigVals = k*eigVals[ind]

    # Store eigenvalues and eigenvectors
    grow[:,cnt] = eigVals[0:Ne].imag
    freq[:,cnt] = eigVals[0:Ne].real
    modes[0,1:N,:,cnt] = eigVecs[0:N-1,0:Ne]
    modes[1,1:N,:,cnt] = eigVecs[N-1:2*N,0:Ne]
    
    print ('Wavenumber (', int(cnt+1), '/', int(Nk),')', ': ',"{:.2f}".format(k),', Growth: ',"{:.4f}".format(grow[0,cnt]),', Phase: ',"{:.4f}".format(freq[0,cnt]))
    
grow[:,:] = comm.reduce(grow[:,:],op=MPI.SUM, root=0)    
freq[:,:] = comm.reduce(freq[:,:],op=MPI.SUM, root=0)    
modes[:,:,:] = comm.reduce(modes[:,:,:],op=MPI.SUM, root=0)    
print("Done!")


if rank==0:
    plt.figure(figsize=(8,6))
    plt.clf()

    # Find max growth rates
    Imax = np.zeros((Ne,1), dtype=int)
    for ii in range(Ne):
        Imax[ii] = np.argmax(grow[ii,:])
        print("Max growth for curve", ii+1, "is", grow[ii,Imax[ii]])
        print("Frequency for curve", ii+1, "is", freq[ii,Imax[ii]])
        #plt.plot(kk[Imax[ii]],grow[ii,Imax[ii]],'o',markersize=10) # mark max growth

    plt.plot(kk,grow[0,:],'.k', )
    plt.plot(kk,grow[1,:],'.k', )
    plt.plot(kk,grow[2,:],'.k', )
    plt.plot(kk,grow[3,:],'.k', )
    plt.grid(True)
    plt.title("Growth Rates (N = "+str(N)+"): "+r"$M^2$ = "+'{:.2f}'.format(M2)+", F = "+'{:.2f}'.format(F))
    #plt.legend(loc='best')
    plt.grid('on')
    plt.xlabel('wavenumber')
    plt.ylabel('growth')
    plt.tight_layout()
    plt.savefig("QGMHD_growth_bickley_spectral_"+str(N)+".png") 
    #plt.show()


    plt.figure(figsize=(12,12))
    plt.clf()

    psi_modes = modes[0,:,:,:]
    mag_modes = modes[1,:,:,:]

    for ii in range(Ne):

        plt.subplot(Ne,2,1+2*ii)
        plt.plot(y,psi_modes[:,ii,Imax[ii]].real,'.-b')
        #plt.xlim(7,13)
        plt.plot(y,psi_modes[:,ii,Imax[ii]].imag,'.-r')
        plt.title('Psi mode '+str(ii+1)+':  growthrate = '+str(grow[ii,Imax[ii]][0]))
        plt.subplot(Ne,2,2+2*ii)
        plt.plot(y,mag_modes[:,ii,Imax[ii]].real,'.-b')
        #plt.xlim(7,13)
        plt.plot(y,mag_modes[:,ii,Imax[ii]].imag,'.-r')
        plt.title('A mode '+str(ii+1)+':  frequency = '+str(freq[ii,Imax[ii]][0]))
        plt.tight_layout()   
    plt.savefig("QGMHD_eigen_bickley_spectral_"+str(N)+".png")     
    #plt.show()

    plt.figure(figsize=(8,8))
    
    for ii in range(Ne):
   
        plt.clf()
 
        Lx = 2*np.pi/kk[Imax[ii]]
        print("2x Domain width: ",2*Lx)
        x = np.linspace(0,8*np.pi,N+1)
        xx,yy = np.meshgrid(x,y)
    
        psi_mode2d = (np.tile(psi_modes[:,ii,Imax[ii]],(1,N+1))*np.exp(2*np.pi*1j*xx/Lx)).real
        mag_mode2d = (np.tile(mag_modes[:,ii,Imax[ii]],(1,N+1))*np.exp(2*np.pi*1j*xx/Lx)).real

        u1_mode2d = -np.gradient(psi_mode2d)[1]
        u2_mode2d = np.gradient(psi_mode2d)[0]
        b1_mode2d = -np.gradient(mag_mode2d)[1]
        b2_mode2d = np.gradient(mag_mode2d)[0]

        q_mode2d = -np.gradient(u1_mode2d)[1]+np.gradient(u2_mode2d)[0]
        j_mode2d = -np.gradient(b1_mode2d)[1]+np.gradient(b2_mode2d)[0]

        plt.subplot(2,2,3)
        plt.pcolormesh(xx,yy,psi_mode2d,cmap='seismic',shading='gouraud')
        plt.colorbar()
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title('Psi mode '+str(ii+1))
        plt.subplot(2,2,4)
        plt.pcolormesh(xx,yy,mag_mode2d,cmap='seismic',shading='gouraud')
        plt.colorbar()
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title('A mode '+str(ii+1))
        plt.subplot(2,2,1)
        plt.pcolormesh(xx,yy,q_mode2d,cmap='seismic',shading='gouraud')
        plt.colorbar()
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title('q mode '+str(ii+1))
        plt.subplot(2,2,2)
        plt.pcolormesh(xx,yy,j_mode2d,cmap='seismic',shading='gouraud')
        plt.colorbar()
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title('j mode '+str(ii+1))

        plt.tight_layout()
        plt.savefig("QGMHD_struct_bickley_mode_"+str(ii+1)+"_spectral_"+str(N)+".png") 
    #plt.show()    



