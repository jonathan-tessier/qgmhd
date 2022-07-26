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

# Physical parameters
L     = 20.                         # Length of domain
N     = 256                         # Number of grid points
F     = 0.0                         # Froude number
M     = 0.0                         # Magnetic number
beta  = 0.0                         # nondim beta from coriolis
Re    = 0.0                         # Reynolds Number
Rm    = 0.0                         # Magnetic Reynolds Number

Reinv = 0.0 # 1/Re       # Inverse Reynolds number to allow for inviscid case
Rminv = 0.0 # 1/Rm       # Inverse Reynolds number to allow for inviscid case

# Jet parameters
Lj = 1.0                           # width of jet
Uj = 1.0                           # maximum velocity of jet

if rank==0:
    print("Computational Parameters:")
    print("Domain Length (L) = "+str(L))
    print("Domain Num Points = "+str(N))
    print("Domain Resolution = "+str(L/N))

## CHEB computes the Chebyshev differentiation matrix
## ------
#    matrix on N+1 points (i.e. N intervals)
#    D = differentiation matrix
#    x = Chebyshev grid

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
## ------

# Define Diff Operators
Dy,y = cheb(N)
y   = (y[:,0]+1)*L/2
Dy  = Dy*(2/L)
Dyy = np.dot(Dy,Dy)

#Define useful Operators
I  = np.identity(N-1)
O  = np.zeros([N-1,N-1])

# Define Basic State
P  = Uj*np.tanh((y-L/2)/Lj)
U  = Uj/(np.cosh((y-L/2)/Lj))**2
Q  = F*P + np.dot(Dyy,P) 
Q2 = -2.*Uj*np.tanh((y-L/2)/Lj)/(np.cosh((y-L/2)/Lj))**2
Uyy= np.dot(Dyy,U) 
B0 = np.ones(np.shape(y)) 
Byy= np.dot(Dyy,B0)

# Define range of parameters
dk = 5e-1; kk = np.arange(dk,2+dk,dk); Nk = len(kk);

#FF = np.linspace(0,1,5); NF = len(FF);
#MM = np.linspace(0,0.1,5); NM = len(MM);
bb = np.linspace(0,1,4); Nb = len(bb);

# Define storage vectors: DIM = [psi, a]x[num modes]x[num wavenumbers]x[num <param>] = 2 x Ne x Nk x N<>
Ne = 4
#c_vals = np.zeros((Ne,Nk),dtype=complex)
grow = np.zeros((Ne,Nk,Nb))
freq = np.zeros((Ne,Nk,Nb))
#modes = np.zeros((2,N+1,Ne,Nk,NF),dtype=complex)

# Loop over <parameter>
p_cnt = 0
for p_cnt in range(Nb):
    # set <parameter>
    beta = bb[p_cnt]
    print ('Parameter Loop: ', int(p_cnt+1), '/', int(Nb))
    # loop over wavenumebrs
    cnt=0
    for cnt in range(rank,len(kk),num_procs):
        k = kk[cnt]
        k2 = k**2
        nabla = Dyy[1:-1,1:-1] - k2*I
        # DIFFUSION
        B = np.vstack((np.hstack((nabla - F*I, O)), np.hstack((O, I))))
        A = np.vstack((np.hstack((np.dot((np.diag(U[1:-1],0)+1j*Reinv/k*(nabla - F*I)),nabla) - np.diag(Uyy[1:-1],0) +beta*I, -M**2*(np.dot(np.diag(B0[1:-1],0),nabla) - np.diag(Byy[1:-1],0)))), np.hstack((-np.diag(B0[1:-1],0), np.diag(U[1:-1],0)+1j*Rminv/k*(nabla)))))  

        # Solve for eigenvalues
        eigVals,eigVecs = spalg.eig(A,B)

        # Sort eigenvalues and eigenvectors
        ind = (-np.imag(eigVals)).argsort()
    
        eigVecs = eigVecs[:,ind]
        eigVals = k*eigVals[ind]

       # Store eigenvalues and eigenvectors
        #c_vals[:,cnt,p_cnt] = eigVals[0:Ne]/k
        grow[:,cnt,p_cnt] = eigVals[0:Ne].imag
        freq[:,cnt,p_cnt] = eigVals[0:Ne].real
        #modes[0,1:N,:,cnt,p_cnt] = eigVecs[0:N-1,0:Ne]
        #modes[1,1:N,:,cnt,p_cnt] = eigVecs[N-1:2*N,0:Ne]
    
        print (' - Wavenumber (', int(cnt+1), '/', int(Nk),')', ': ',"{:.2f}".format(k),', Growth: ',"{:.4f}".format(grow[0,cnt,p_cnt]),', Phase: ',"{:.4f}".format(freq[0,cnt,p_cnt]))
        cnt += 1
    grow[:,:,p_cnt] = comm.reduce(grow[:,:,p_cnt],op=MPI.SUM, root=0)    
    freq[:,:,p_cnt] = comm.reduce(freq[:,:,p_cnt],op=MPI.SUM, root=0)         
    p_cnt+=1
print("Done!")

if rank==0:
    for which_mode in range(2):
        # Plot the two most unstable modes
        plt.figure(figsize=(10,10))
        plt.clf()
        dataset = np.transpose(grow[which_mode,:,:])
        kplot,p_plot = np.meshgrid(kk,bb)
        plt.contourf(kplot,p_plot,dataset,cmap='Greys',levels=10) 
        plt.title("Mode "+str(which_mode+1)+": Growth rate, Im(omega) for F = "+str(F)+", M = "+str(M))
        plt.xlabel("k")
        plt.ylabel("beta")
        #plt.clim([0,1])
        plt.colorbar()
        plt.savefig("QGMHD_growth_betacontour_Mode"+str(which_mode+1)+".png") 
        #plt.show()

