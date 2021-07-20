import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys

import subprocess

def merge_to_mp4(frame_filenames, movie_name, fps=12):
    f_log = open("ffmpeg.log", "w")
    f_err = open("ffmpeg.err", "w")
    cmd = ['ffmpeg', '-framerate', str(fps), '-i', frame_filenames, '-y', 
            '-q', '1', '-threads', '0', '-pix_fmt', 'yuv420p', movie_name]
    subprocess.call(cmd, stdout=f_log, stderr=f_err)
    f_log.close()
    f_err.close()

def plot_PV_A(q, j, x, y, cnt, t, movie, fig, axes):

    dx, dy = x[1] - x[0],  y[1] - y[0]
    Lx, Ly = x[-1] + dx/2, y[-1] + dy/2

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim((0, Lx))
        ax.set_ylim((0, Ly))
        ax.set_xticks([0, Lx/2, Lx])
        ax.set_xticklabels([0, Lx/2, Lx], fontsize=16)
        ax.set_yticks([0, Ly/2, Ly])
        ax.set_yticklabels([0, Ly/2, Ly], fontsize=16)
     
    axes[0].set_title('PV at t = %4.2f' % t, fontsize=20)
    axes[1].set_title('j at t = %4.2f' % t, fontsize=20)
    color0 = axes[0].pcolormesh(x, y, q, cmap="coolwarm", shading = 'gouraud')
    color1 = axes[1].pcolormesh(x, y, j, cmap="coolwarm", shading = 'gouraud')
    fig.tight_layout()

    plt.draw()
    plt.pause(0.0001)

    if movie:
        plt.savefig('frame_{0:04d}.png'.format(ii), dpi=200)

M = 7
N = (2**M, 2**M)

folder           = 'output-qgmhd/'
file_name = folder + "qgmhd_Nx{}_diagnostics.h5".format(N[0])
filenc    = folder + 'qgmhd_Nx{}_variables.nc'.format(N[0])

f = Dataset(filenc, 'r', format='NETCDF4')

#print(f)
#print(f.variables.keys())
#tt = f.variables['Time']

tp = f.variables['TimePlot']
x  = f.variables['x']
y  = f.variables['y']

Q = f.variables['PV']
#A = f.variables['A']
j = f.variables['j']
#u = f.variables['u']
#v = f.variables['v']
#b1= f.variables['b1']
#b2= f.variables['b2']

movie = False
movie_name = 'qgmhd_1L_movie.mp4'

fig, axes = plt.subplots(ncols=2, figsize=(12,6) )

for ii in range(len(tp)):

    t = tp[ii]

    plot_PV_A(Q[ii], j[ii], x, y, ii, t, movie, fig, axes)

if movie:
    merge_to_mp4('frame_%04d.png', movie_name)




