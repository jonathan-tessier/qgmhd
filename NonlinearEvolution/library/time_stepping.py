import numpy as np

# Set up params stuff        
class temporal_parameters(object):

    def __init__(
            self,

            t0    = 0.0,
            tf    = 1.0,
            dt    = 1e-2,
            tplot = 1.0
    ):

        """
        Temporal Parameters 
        """ 

        self.t0    = t0
        self.tf    = tf
        self.dt    = dt
        self.tplot = tplot
        
        self.Nt    = int(self.tf/self.dt) + 2
        self.npt   = int(self.tplot/dt)
        self.tt    = np.arange(self.Nt)*dt
        self.ttplt = np.arange(int((self.Nt-1)/self.npt)+1)*dt*self.npt

        
from library.qg_physics import flux_qgmhd, spectral_filter, output_diagnostics

# time stepping stuff
def solve_model(phys, times, soln, soln_bar, gr_q, gr_A, gr_j, gr_u, gr_v, gr_b1, gr_b2, diagvals):

    # Initialize fields
    Nx, Ny = soln.grid.Nx, soln.grid.Ny
            
    ### Euler step
    cnt = 0
    t   = times.t0 + cnt*times.dt

    if cnt % times.npt == 0:
        cnt_nc = int(cnt/times.npt)
        gr_q[cnt_nc], gr_A[cnt_nc], gr_j[cnt_nc]  = soln.q, soln.A, soln.j
        gr_u[cnt_nc], gr_v[cnt_nc], gr_b1[cnt_nc], gr_b2[cnt_nc] = soln.u[0], soln.u[1], soln.b[0], soln.b[1]
    
    NLnm, diagvals[cnt,:] = flux_qgmhd(soln, soln_bar, phys)
    
    soln.q = soln.q + times.dt*NLnm[0]  
    soln.A = soln.A + times.dt*NLnm[1]  

    output_diagnostics(diagvals, cnt, t);
    spectral_filter(soln)

    ### AB2 step
    cnt = 1
    t   = times.t0 + cnt*times.dt
    
    if cnt % times.npt == 0:
        cnt_nc = int(cnt/times.npt)
        gr_q[cnt_nc], gr_A[cnt_nc], gr_j[cnt_nc] = soln.q, soln.A, soln.j   
        gr_u[cnt_nc], gr_v[cnt_nc], gr_b1[cnt_nc], gr_b2[cnt_nc] = soln.u[0]-soln_bar.u[0], soln.u[1]-soln_bar.u[1], soln.b[0]-soln_bar.b[0], soln.b[1]-soln_bar.b[1]

    NLn, diagvals[cnt,:] = flux_qgmhd(soln, soln_bar, phys)

    soln.q = soln.q + 0.5*times.dt*(3.*NLn[0] - NLnm[0])  
    soln.A = soln.A + 0.5*times.dt*(3.*NLn[1] - NLnm[1])  

    output_diagnostics(diagvals, cnt, t);
    spectral_filter(soln)

    # AB3 step
    for cnt in range(2,times.Nt):

        t   = times.t0 + cnt*times.dt
        
        NL, diagvals[cnt,:] = flux_qgmhd(soln, soln_bar, phys)
        
        soln.q  = soln.q + times.dt/12.*(23*NL[0] - 16.*NLn[0] + 5.*NLnm[0]).real
        soln.A  = soln.A + times.dt/12.*(23*NL[1] - 16.*NLn[1] + 5.*NLnm[1]).real

        output_diagnostics(diagvals, cnt, t);
        spectral_filter(soln)

        # Reset fluxes
        NLnm = NLn
        NLn  = NL

        if cnt % times.npt == 0:
            cnt_nc = int(cnt/times.npt)
            gr_q[cnt_nc], gr_A[cnt_nc], gr_j[cnt_nc] = soln.q, soln.A, soln.j
            gr_u[cnt_nc], gr_v[cnt_nc], gr_b1[cnt_nc], gr_b2[cnt_nc] = soln.u[0]-soln_bar.u[0], soln.u[1]-soln_bar.u[1], soln.b[0]-soln_bar.b[0], soln.b[1]-soln_bar.b[1]


