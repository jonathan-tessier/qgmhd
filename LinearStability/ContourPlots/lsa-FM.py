import numpy as np
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import sys
import h5py

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
L     = 8*np.pi                     # Length of domain
N     = 512                         # Number of grid points
F     = 0.0                         # Froude number
M     = 0.0                         # Magnetic number
beta  = 0.0                         # nondim beta from coriolis
Re    = 1e4                         # Reynolds Number
Rm    = 1e4                         # Magnetic Reynolds Number

k     = 0.5                         # Wavenumber

Reinv = 1/Re                        # Inverse Reynolds number to allow for inviscid case
Rminv = 1/Rm                        # Inverse Reynolds number to allow for inviscid case

# Jet parameters
Lj = 1.0                            # width of jet
Uj = 1.0                            # maximum velocity of jet

# save dataset
save2h5 = True

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
#dk = 5e-2; kk = np.arange(dk,2+dk,dk); Nk = len(kk);
FF = np.linspace(0,10,100); NF = len(FF);
MM = np.linspace(0,0.5,100); NM = len(MM);
#bb = np.linspace(0,1,5); Nb = len(bb);

# Define storage vectors: DIM = [psi, a]x[num modes]x[num wavenumbers]x[num <param>] = 2 x Ne x Nk x N<>
Ne = 4
#c_vals = np.zeros((Ne,Nk),dtype=complex)
grow = np.zeros((Ne,NM,NF))
freq = np.zeros((Ne,NM,NF))
p_modes = np.zeros((N+1,Ne,NM,NF),dtype=complex)
a_modes = np.zeros((N+1,Ne,NM,NF),dtype=complex)

# Loop over <parameter>
p_cnt = 0
for p_cnt in range(NF):
    # set <parameter>
    F = FF[p_cnt]
    print ('F Loop: ', int(p_cnt+1), '/', int(NF))
    # loop over wavenumebrs
    cnt=0
    for cnt in range(rank,len(MM),num_procs):
        M = MM[cnt]
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
        p_modes[1:N,:,cnt,p_cnt] = eigVecs[0:N-1,0:Ne]
        a_modes[1:N,:,cnt,p_cnt] = eigVecs[N-1:2*N,0:Ne]
    
        print (' - M Loop: (', int(cnt+1), '/', int(NM),')', ': ',"{:.2f}".format(M),', Growth: ',"{:.4f}".format(grow[0,cnt,p_cnt]),', Phase: ',"{:.4f}".format(freq[0,cnt,p_cnt]))
        
    grow[:,:,p_cnt] = comm.reduce(grow[:,:,p_cnt],op=MPI.SUM, root=0)    
    freq[:,:,p_cnt] = comm.reduce(freq[:,:,p_cnt],op=MPI.SUM, root=0)
    p_modes[:,:,:,p_cnt] = comm.reduce(p_modes[:,:,:,p_cnt],op=MPI.SUM, root=0)
    a_modes[:,:,:,p_cnt] = comm.reduce(a_modes[:,:,:,p_cnt],op=MPI.SUM, root=0)
    p_cnt+=1

print("Done!")

# Output parameters for pickup files in HDF5
if save2h5 and rank==0:
    filename = "output.h5"                        # output filename
    file = h5py.File(filename, mode="w")          # output file

    # save parameters
    file.create_dataset("L",   data = L)      
    file.create_dataset("N",   data = N) 
    file.create_dataset("Ne",  data = Ne)
    file.create_dataset("F",   data = FF)    
    file.create_dataset("M2",  data = MM**2)          
    file.create_dataset("kk",  data = k)      
    file.create_dataset("Re",  data = Re)
    file.create_dataset("Rm",  data = Rm)
    file.create_dataset("beta",data = beta)
    file.create_dataset("Lj",  data = Lj)
    file.create_dataset("Uj",  data = Uj)

    # Output Fields
    output_grow  = file.create_dataset('grow', data=grow) # growth rate
    output_freq  = file.create_dataset('freq', data=freq) # frequency
    output_pmode = file.create_dataset('p_mode', data=p_modes) # Psi Modes
    output_amode = file.create_dataset('a_mode', data=a_modes) # A Modes

    print("File "+filename+" opened")

if rank==0:
    for which_mode in range(2):
        # Plot the two most unstable modes
        plt.figure(figsize=(10,10))
        plt.clf()
        dataset = np.transpose(grow[which_mode,:,:])
        x_plot,y_plot = np.meshgrid(MM,FF)
        plt.contourf(x_plot,y_plot,dataset,cmap='Greys',levels=10) 
        plt.title("Mode "+str(which_mode+1)+": Growth rate, Im(omega) for k = "+str(k)+", beta = "+str(beta))
        plt.xlabel("M")
        plt.ylabel("F")
        #plt.clim([0,1])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("QGMHD_FMcontour_Mode"+str(which_mode+1)+".png") 
        #plt.show()
