import numpy as np

class init_solution(object):

    def __init__(
            self, 
            grid
    ):

        self.grid = grid
        Nx, Ny = 2**grid.M,  2**grid.M

        self.q     = np.zeros((Ny,Nx))
        self.psi   = np.zeros((Ny,Nx))
        self.j     = np.zeros((Ny,Nx))
        self.A     = np.zeros((Ny,Nx))
        self.fq    = np.zeros((Ny,Nx))
        self.fA    = np.zeros((Ny,Nx))
        self.u     = np.zeros((2,Ny,Nx))
        self.b     = np.zeros((2,Ny,Nx))
        self.gradq = np.zeros((2,Ny,Nx))
        self.gradj = np.zeros((2,Ny,Nx))
        
