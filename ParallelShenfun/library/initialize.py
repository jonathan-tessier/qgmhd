import numpy as np
from shenfun import *
from library.grid import grid_parameters, generate_field_with_kpeak, generate_broad_spectrum_field
from library.operators import inner
import sys
import matplotlib.pyplot as plt

class init_solution(object):

    def __init__(
            self, 
            grid,
            T,
            TV,
            VM
    ):

        self.grid = grid
        Nx, Ny = grid.Nx,  grid.Ny
        N = (Nx, Ny)
        d = ((0., grid.Lx), (0., grid.Ly))

        self.qA, self.qA_hat = Array(VM), Function(VM)
        self.psi = Array(T)
        self.j  = Array(T)
        self.A  = Array(T)
        self.u  = Array(TV)
        self.b  = Array(TV)
        self.gradq = Array(TV)
        self.gradj = Array(TV)
        self.FqA = Array(VM)

def Build_FunctionSpaces(Nx, Ny, Lx, Ly):
    
    dim = 2

    V1  = FunctionSpace(Nx, 'F', dtype='D', domain=(0, Lx))
    V2  = FunctionSpace(Ny, 'F', dtype='d', domain=(0, Ly))
    T   = TensorProductSpace(comm, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
    T0  = TensorProductSpace(MPI.COMM_SELF, (V1, V2), **{'planner_effort': 'FFTW_MEASURE'})
    TV  = VectorSpace(T)
    VM  = CompositeSpace([T]*dim)

    # Try to import wisdom.
    try:
        fftw.import_wisdom('QG1L.wisdom')
        print('Importing wisdom')
    except:
        print('No wisdom imported')

    return T0, T, TV, VM

def scatter_ICs(T0, grid, phys, comm, rank, size, ICtype = "broad_random"):

    print("Opened Scatter function")
    gridS  = grid_parameters(phys, T0, Nx = grid.Nx, Ny = grid.Ny, Lx = grid.Lx, Ly = grid.Ly)

    X = gridS.X; 
    Xn= [gridS.X[0]/grid.L[0],gridS.X[1]/grid.L[1]] # normalized coordinates
    Xs= [gridS.X[0]-grid.L[0]/2,gridS.X[1]-grid.L[1]/2] # shifted coordinates

    vortex_factor2 = -1/(2*np.pi/grid.L[0])
    vortex_factor4 = -1/(4*np.pi/grid.L[0])

    if rank == 0:

        # initialize required fields
        K2F2    = gridS.K2F2
        K2      = gridS.K2
        iK      = gridS.iK
        TV      = VectorSpace(T0)
        u0      = Array(TV)
        b0      = Array(TV)
        q0      = Array(T0)
        p0      = Array(T0)
        j0      = Array(T0)
        A0      = Array(T0)

        if ICtype == "broad_random":

            p0  = generate_broad_spectrum_field(T0, size = np.shape(gridS.X[0]), grid = gridS, \
                  kstar = phys.psikstar, sigma = phys.psi_sig, npseed = phys.psi_nps, amp=phys.psi_amp)
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            A0  = generate_broad_spectrum_field(T0, size = np.shape(gridS.X[0]), grid = gridS, \
                  kstar = phys.Akstar, sigma = phys.A_sig, npseed = phys.A_nps, amp=phys.A_amp)
            j0  = T0.backward(-K2*T0.forward(A0),j0)

            for i in range(2):
                ip = (i + 1) % 2
                u0[i] = T0.backward(-(-1)**i*iK[ip]*T0.forward(p0), u0[i])
                b0[i] = T0.backward(-(-1)**i*iK[ip]*T0.forward(A0), b0[i])

            print("Initial Random Field Diagnostics")
            print("===============")
            rms0 = np.sqrt(np.mean(inner(u0,u0)))
            sca0 = np.amax(np.sqrt(inner(u0,u0)))
            ani0 = np.mean(u0[0]**2)/np.mean(u0[0]**2+u0[1]**2)
            KE = 0.5*np.mean(inner(u0,u0))*gridS.Lx*gridS.Ly
            print("rms(u) = "+str(rms0))
            print("ani(u) = "+str(ani0))
            print("sca(u) = "+str(sca0))
            print("KE(0)  = "+str(KE))

            rms1 = np.sqrt(phys.M2*np.mean(inner(b0,b0)))
            ani1 = np.mean(b0[0]**2)/np.mean(b0[0]**2+b0[1]**2)
            sca1 = np.amax(np.sqrt(inner(b0,b0)))
            ME = 0.5*np.mean(phys.M2*inner(b0,b0))*gridS.Lx*gridS.Ly
            print("rms(b) = "+str(rms1))
            print("ani(b) = "+str(ani1))
            print("sca(b) = "+str(sca1))
            print("ME(0)  = "+str(ME))
            print()

        elif ICtype == "peak_random":

            p0  = generate_field_with_kpeak(T0, size = np.shape(gridS.X[0]), grid = gridS, \
                  kstar = phys.psikstar, sigma = phys.psi_sig, npseed = phys.psi_nps, amp=phys.psi_amp)
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            A0  = generate_field_with_kpeak(T0, size = np.shape(gridS.X[0]), grid = gridS, \
                  kstar = phys.Akstar, sigma = phys.A_sig, npseed = phys.A_nps, amp=phys.A_amp)
            j0  = T0.backward(-K2*T0.forward(A0),j0)

            for i in range(2):
                ip = (i + 1) % 2
                u0[i] = T0.backward(-(-1)**i*iK[ip]*T0.forward(p0), u0[i])
                b0[i] = T0.backward(-(-1)**i*iK[ip]*T0.forward(A0), b0[i])

            print("Initial Random Field Diagnostics")
            print("===============")
            rms0 = np.sqrt(np.mean(inner(u0,u0)))
            sca0 = np.amax(np.sqrt(inner(u0,u0)))
            ani0 = np.mean(u0[0]**2)/np.mean(u0[0]**2+u0[1]**2)
            KE = 0.5*np.mean(inner(u0,u0))*gridS.Lx*gridS.Ly
            print("rms(u) = "+str(rms0))
            print("ani(u) = "+str(ani0))
            print("sca(u) = "+str(sca0))
            print("KE(0)  = "+str(KE))

            rms1 = np.sqrt(phys.M2*np.mean(inner(b0,b0)))
            ani1 = np.mean(b0[0]**2)/np.mean(b0[0]**2+b0[1]**2)
            sca1 = np.amax(np.sqrt(inner(b0,b0)))
            ME = 0.5*np.mean(phys.M2*inner(b0,b0))*gridS.Lx*gridS.Ly
            print("rms(b) = "+str(rms1))
            print("ani(b) = "+str(ani1))
            print("sca(b) = "+str(sca1))
            print("ME(0)  = "+str(ME))
            print()

        elif ICtype == "big_vortex":
            xmin, xmax, ymin, ymax = 1*grid.Lx/4, 3*grid.Lx/4, 1*grid.Ly/4, 3*grid.Ly/4
            window = (xmin <= X[0])*(xmax >= X[0])*(ymin <= X[1])*(ymax >= X[1])
            p0  = phys.psi_amp*vortex_factor2*np.cos(2*np.pi*Xn[0])*np.cos(2*np.pi*Xn[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            p0  = np.multiply(p0, window.astype(float))
            q0  = np.multiply(q0, window.astype(float))
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "single_vortex":
            xmin, xmax, ymin, ymax = 3*grid.Lx/8, 5*grid.Lx/8, 3*grid.Ly/8, 5*grid.Ly/8
            window = (xmin <= X[0])*(xmax >= X[0])*(ymin <= X[1])*(ymax >= X[1])
            p0  = phys.psi_amp*vortex_factor4*np.cos(4*np.pi*Xn[0])*np.cos(4*np.pi*Xn[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            p0  = np.multiply(p0, window.astype(float))
            q0  = np.multiply(q0, window.astype(float))
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "quad_vortex":
            p0  = -phys.psi_amp*vortex_factor2*np.sin(2*np.pi*Xn[0])*np.sin(2*np.pi*Xn[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "row_vortex":
            ymin, ymax = 3*grid.Ly/8, 5*grid.Ly/8
            window = (ymin <= X[1])*(ymax >= X[1])
            p0  = -phys.psi_amp*vortex_factor4*np.sin(4*np.pi*Xn[0])*np.cos(4*np.pi*Xn[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            p0  = np.multiply(p0, window.astype(float))
            q0  = np.multiply(q0, window.astype(float))
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "col_vortex":
            xmin, xmax = 3*grid.Lx/8, 5*grid.Lx/8
            window = (xmin <= X[0])*(xmax >= X[0])
            p0  = phys.psi_amp*vortex_factor4*np.sin(4*np.pi*Xn[1])*np.cos(4*np.pi*Xn[0])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            p0  = np.multiply(p0, window.astype(float))
            q0  = np.multiply(q0, window.astype(float))
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "weiss_isolated":
            xmin, xmax, ymin, ymax = grid.Lx/4, 3*grid.Lx/4, grid.Ly/4, 3*grid.Ly/4
            window = (xmin <= X[0])*(xmax >= X[0])*(ymin <= X[1])*(ymax >= X[1])
            p0  = (phys.psi_amp/(2*np.pi))*np.cos(2*np.pi*Xs[0])*np.cos(2*np.pi*Xs[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            p0  = np.multiply(p0, window.astype(float))
            q0  = np.multiply(q0, window.astype(float))
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "weiss_single":

            p0  = (phys.psi_amp/np.pi)*(1-4*Xs[0]**2)**4*np.cos(np.pi*Xs[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "weiss_band":

            p0  = -(phys.psi_amp/(4*np.pi))*(1-4*Xs[0]**2)**4*np.sin(4*np.pi*Xs[1])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "weiss_doubleband":
            xmin, xmax = grid.Lx/4, 3*grid.Lx/4
            window = (xmin <= X[0])*(xmax >= X[0])
            p0  = (phys.psi_amp/(4*np.pi))*np.sin(2*np.pi*Xs[1])*np.sin(4*np.pi*Xs[0])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            p0  = np.multiply(p0, window.astype(float))
            q0  = np.multiply(q0, window.astype(float))
            j0  = 0*q0
            A0  = 0*q0

        elif ICtype == "weiss_hband":

            p0  = (phys.psi_amp/(4*np.pi))*(1-4*Xs[1]**2)**4*np.sin(4*np.pi*Xs[0])
            q0  = T0.backward(-K2F2*T0.forward(p0),q0)
            j0  = 0*q0
            A0  = 0*q0

        else:
            raise(NameError("Perturbation IC name not recognized"))

        q_split = np.array_split(q0, size, axis = 0)
        p_split = np.array_split(p0, size, axis = 0)
        j_split = np.array_split(j0, size, axis = 0)
        A_split = np.array_split(A0, size, axis = 0)

    else:
        q0 = None
        p0 = None
        j0 = None
        A0 = None
        q_split = None
        p_split = None
        j_split = None
        A_split = None

    q0 = comm.bcast(q0,root=0)
    p0 = comm.bcast(p0,root=0)
    j0 = comm.bcast(j0,root=0)
    A0 = comm.bcast(A0,root=0)

    q_split = comm.bcast(q_split,root=0)
    p_split = comm.bcast(p_split,root=0)
    j_split = comm.bcast(j_split,root=0)
    A_split = comm.bcast(A_split,root=0)

    return q_split[rank], p_split[rank], j_split[rank], A_split[rank]

def scatter_forcing(T0, grid, phys, comm, rank, size):

    print("Opened Scatter function")
    gridS  = grid_parameters(phys, T0, Nx = grid.Nx, Ny = grid.Ny, Lx = grid.Lx, Ly = grid.Ly)

    if rank == 0:
        qF  = generate_field_with_kpeak(T0, size = np.shape(gridS.X[0]), grid = gridS, \
              kstar = phys.Fqkstar, sigma = phys.Fq_sig, npseed = phys.Fq_nps, amp=phys.Fq_amp)
        AF  = generate_field_with_kpeak(T0, size = np.shape(gridS.X[0]), grid = gridS, \
              kstar = phys.FAkstar, sigma = phys.FA_sig, npseed = phys.FA_nps, amp=phys.FA_amp)
        qF_split = np.array_split(qF, size, axis = 0)
        AF_split = np.array_split(AF, size, axis = 0)

    else:
        qF = None
        AF = None
        qF_split = None
        AF_split = None

    qF = comm.bcast(qF,root=0)
    AF = comm.bcast(AF,root=0)
    qF_split = comm.bcast(qF_split,root=0)
    AF_split = comm.bcast(AF_split,root=0)

    return qF_split[rank], AF_split[rank]


