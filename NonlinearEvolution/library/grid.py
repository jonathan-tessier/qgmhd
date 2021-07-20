import numpy as np

# Set up params stuff        
class grid_parameters(object):

    def __init__(
            self,
            phys,
            M  = 6,
            Lx = 20.0,
            Ly = 20.0
    ):

        """
        Grid Parameters
        """

        self.phys = phys
        
        self.M  = M
        self.Nx = 2**M
        self.Ny = 2**M
        self.sc = 2**(M-7)
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx/self.Nx
        self.dy = Ly/self.Ny

        self.domain = ((0., Ly), (0., Lx))
    
        self.x = np.linspace(self.dx/2, Lx-self.dx/2, self.Nx)
        self.y = np.linspace(self.dy/2, Ly-self.dy/2, self.Ny)
        
        self.xx,self.yy = np.meshgrid(self.x, self.y)
        
        self.kx = 2*np.pi/Lx*np.hstack([range(0,int(self.Nx/2)+1), range(-int(self.Nx/2)+1,0)])
        self.ky = 2*np.pi/Ly*np.hstack([range(0,int(self.Ny/2)+1), range(-int(self.Ny/2)+1,0)])
        kxx, kyy = np.meshgrid(self.kx, self.ky)

        self.iK     = np.zeros((2,self.Ny,self.Nx), dtype=complex)
        self.iKoK2  = np.zeros((2,self.Ny,self.Nx), dtype=complex)
        
        self.iK[0]  = 1j*kxx
        self.iK[1]  = 1j*kyy
        self.K2     = kxx**2 + kyy**2
        self.K2F2   = self.K2 + self.phys.F2
        self.K2F2inv  = np.where(self.K2F2 == 0.0, 0.0, 1.0/self.K2F2).astype(float)
        self.iKoK2F2  = self.iK*self.K2F2inv
        #self.iKoK2[1] = self.iK[1]*self.K2inv

        # Filter Parameters (Smooth)
        kmax = max(abs(self.kx)); 
        ks = 0.8*kmax;
        km = 0.9*kmax;
        alpha = 0.69*ks**(-1.88/np.log(km/ks));
        beta  = 1.88/np.log(km/ks);
        self.sfilt = np.exp(-alpha*(kxx**2 + kyy**2)**(beta/2.0));
        # adding option for a 2/3 filter (Sharp)
        #kfilter = kmax*2/3
        #self.sfilt = (np.sqrt(kxx**2 + kyy**2)<=kfilter).astype(float)


from scipy.fftpack import fft, ifft, fftn, ifftn
from scipy import interpolate

# random field generation: Annulus in spectral space about kstar with width sigma 
def generate_field_with_kpeak(grid,kstar,sigma,npseed,amp):

    kxx, kyy = np.meshgrid(grid.kx, grid.ky)
    krr = np.sqrt(kxx**2  + kyy**2)
 
    dk = grid.kx[2]-grid.kx[1]
    kmin, kmax  = kstar - sigma/2, kstar + sigma/2
    window = (kmin <= krr) * (kmax >= krr)

    np.random.seed(npseed)
    fhat = np.multiply(fftn(np.random.uniform(-1., 1., size=(grid.Nx, grid.Ny))), window.astype(float))
    f = ifftn(fhat).real
    f = f-np.mean(f)
    f = amp*f/np.max(abs(f))

    return f

# random field generation: Gaussian bump in spectral space (see window definition)
def generate_broad_spectrum_field(grid,kstar,sigma,npseed,amp):

    kxx, kyy = np.meshgrid(grid.kx, grid.ky)
    krr = np.sqrt(kxx**2  + kyy**2)

    window = np.exp(-(krr/(sigma*kstar))**2)

    np.random.seed(npseed)
    fhat = np.multiply(fftn(np.random.uniform(-1., 1., size=(grid.Nx, grid.Ny))), window.astype(float))
    f = ifftn(fhat).real
    f = f-np.mean(f)
    f = amp*f/np.max(abs(f))

    return f
