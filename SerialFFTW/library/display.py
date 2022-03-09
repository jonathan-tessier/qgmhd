
# Set up params stuff
def display_parameters(phys, times, grid):

    print(' ')
    print('Physical Parameters')
    print('===================')
    print('F2    = ', phys.F2, '\nM2    = ', phys.M2, \
          '\nRm    = ', phys.Rm, '\nRe    = ', phys.Re, \
          '\nUj    = ', phys.Uj, '\namp   = ', phys.amp, \
          '\nstate = ', phys.state)
    print(' ')
    print('Temporal Parameters')
    print('===================')
    print('t0  = ', times.t0, '\ntf  = ', times.tf, '\ndt  = ', times.dt, \
          '\nNt  = ', times.Nt)
    print(' ')
    print('Grid Parameters')
    print('===============')
    print('Nx  = ', grid.Nx, '\nNy  = ', grid.Ny, \
          '\nLx  = ', grid.Lx, '\nLy  = ', grid.Ly)
    print(' ')

