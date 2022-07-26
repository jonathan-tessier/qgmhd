#!/usr/bin/env python

# post-processing script to read the diagnostics output from 
# multiple runs of the qgmhd_shenfun.py driver script to plot
# diagnostics as a function of M and time (mag. field strength)
#
# Current outputs: energy, anisotropy, lorentz force, microscales
# growth rates, conserved quantities 
#
# In the same directory as this script, create folders named as
# <case_prefix>-M2-<mag_value> Eg: turbulence-M2-1em4, which 
# each contain a 'output-qgmhd' output folder from the model.
# Specify the case prefix below which is fixed for a set of 
# figure. Further, list the values of magnetism to include,
# so the suffixes to your directory names, in mag_values.
# The code will extract the actual value of M from each output.
#
# This scripts further requires a symbolic link of the code library.
# from this directory, run $ ln -s /path/to/library

#imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from library.data_output import fmt, panelindex

# set resolution to pick out correct files
N = [1024]*2

# set the directories you want to read from
case_prefix = "jetB0-F2-0"

# temporal range for growth rate computation
t_bound = [30,38]

# legend
lg = False

# linestyles
color_index = ['k','r','g','b']
lines_index = [(0,()), (0,(3,3)), (0,(3,1,1,1)), (0,(1,1))]
#lines_index = ['-','--','-.',':']
lw = 2.5

# list of directory suffixes corresponding to M2 values 
#mag_values = ["hydro","1em6","1em5","1em4","1em3","1em2","1em1","1em0"]
mag_values = ["hydro","1em6","1em4","1em2"]

figsize=(4.15,4) # actual figure size
bspace = 0.17

# fig format (eps,png,both)
save_format = "png"
dpi=300

filename_array = [] # init dir

# create array of directory files to open
for magval in mag_values:
    filename_array.append(case_prefix+"-M2-"+magval+\
        "/output-qgmhd/qgmhd_Nx{}_realdiagnostics.h5".format(N[0]))

# initialize and show all files
print('Read from files \n', filename_array)
file_array = []; M2_vector = [] # init dirs
KE_vector    = []; 
PE_vector    = []; 
Q2_vector    = []; 
Q1_vector    = []; 
qgrow_vector = []; 
ME_vector    = []; 
A2_vector    = []; 
A1_vector    = []; 
j2_vector    = []; 
Agrow_vector = []; 
Hel_vector   = []; 
TE_vector    = []; 
Lu_vector    = []; 
Lb_vector    = []; 
u12_vector   = [];
u22_vector   = [];
b12_vector   = [];
b22_vector   = [];
qx2_vector   = [];
qy2_vector   = [];
jx2_vector   = [];
jy2_vector   = [];
lor_vector   = [];
lpp_vector   = [];

# first file for invariant quantities
file0 = h5py.File(filename_array[0], "r")
times  = file0['times']
tt     = np.arange(times[0], times[1], times[2])
domain = file0['domain']
N      = file0['size']
F2     = file0['F2']
Uj     = file0['Uj']

print('Opening files...')

# for 0/0 current diagnostics
np.seterr(divide='ignore', invalid='ignore')

# looping through files, pick out the M value and store the diagnostics
for filename in filename_array:
    file1 = h5py.File(filename, "r")
    file_array.append(file1)
    M2_vector.append(float(file1['M2'][...]))
    vals = file1['values']

    KE_vector.append(vals[:,0])
    PE_vector.append(vals[:,1])
    Q2_vector.append(vals[:,2])
    Q1_vector.append(vals[:,3])
    qgrow_vector.append(vals[:,4])
    ME_vector.append(vals[:,5])
    A2_vector.append(vals[:,6])
    A1_vector.append(vals[:,7])
    j2_vector.append(vals[:,8])
    Agrow_vector.append(vals[:,9])
    Hel_vector.append(vals[:,10])
    TE_vector.append(vals[:,0]+vals[:,1]+vals[:,5])
    u12_vector.append(vals[:,11])
    u22_vector.append(vals[:,12])
    b12_vector.append(vals[:,13])
    b22_vector.append(vals[:,14])
    qx2_vector.append(vals[:,15])
    qy2_vector.append(vals[:,16])
    jx2_vector.append(vals[:,17])
    jy2_vector.append(vals[:,18])
    lor_vector.append(vals[:,19])
    lpp_vector.append(vals[:,21])
    Lu_vector.append( np.sqrt(vals[:,11]+vals[:,12])/np.sqrt(vals[:,21]))
    Lb_vector.append( np.sqrt(vals[:,13]+vals[:,14])/np.sqrt(vals[:,8]))

# show M values
print(M2_vector)

# Now for the plotting

### Energy

fig_TE, axesTE = plt.subplots(nrows=1, ncols=1,figsize=figsize)
#fig_TE.suptitle(r'$E = \frac{1}{2} \iint {\bf u\cdot u} + F\psi\cdot\psi + M^2 {\bf b\cdot b} dxdy$')
fig_KE, axesKE = plt.subplots(nrows=1, ncols=1,figsize=figsize)
#fig_KE.suptitle(r'$E_K = \frac{1}{2} \iint {\bf u\cdot u} dxdy$')
fig_PE, axesPE = plt.subplots(nrows=1, ncols=1,figsize=figsize)
#fig_PE.suptitle(r'$E_P = \frac{F}{2} \iint \psi\cdot\psi dxdy$')
fig_ME, axesME = plt.subplots(nrows=1, ncols=1,figsize=figsize)
#fig_ME.suptitle(r'$E_M = \frac{M^2}{2} \iint {\bf b\cdot b } dxdy$')

for index in range(len(mag_values)):
    axesTE.plot(tt, TE_vector[index][:-1], lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
    axesKE.plot(tt, KE_vector[index][:-1], lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
    axesPE.plot(tt, PE_vector[index][:-1], lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
    axesME.plot(tt, ME_vector[index][:-1], lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))

for axes in [axesTE,axesKE,axesPE,axesME]:    
    axes.grid(True)
    axes.set_xlabel("t");

axesTE.set_ylabel(r'$E$')  # = \frac{1}{2} \iint {\bf u\cdot u} + F\psi\cdot\psi + M^2 {\bf b\cdot b} dxdy$')
axesKE.set_ylabel(r'$E_K$')# = \frac{1}{2} \iint {\bf u\cdot u} dxdy$')
axesPE.set_ylabel(r'$E_P$')# = \frac{F}{2} \iint \psi\cdot\psi dxdy$')
axesME.set_ylabel(r'$E_M$')# = \frac{M^2}{2} \iint {\bf b\cdot b } dxdy$')

for fig in [fig_TE, fig_PE, fig_KE, fig_ME]:
    if lg: 
        fig.legend( loc='lower center',fancybox=True, shadow=True, ncol=4)
        fig.subplots_adjust(bottom=bspace)
    fig.tight_layout()

if save_format == 'png' or save_format == 'both':
    fig_TE.savefig("qgmhd_TEdiag_N{}.png".format(N[0]),dpi=dpi)
    fig_PE.savefig("qgmhd_PEdiag_N{}.png".format(N[0]),dpi=dpi)
    fig_KE.savefig("qgmhd_KEdiag_N{}.png".format(N[0]),dpi=dpi)
    fig_ME.savefig("qgmhd_MEdiag_N{}.png".format(N[0]),dpi=dpi)
if save_format == 'eps' or save_format == 'both':
    fig_TE.savefig("qgmhd_TEdiag_N{}.eps".format(N[0]))
    fig_PE.savefig("qgmhd_PEdiag_N{}.eps".format(N[0]))
    fig_KE.savefig("qgmhd_KEdiag_N{}.eps".format(N[0]))
    fig_ME.savefig("qgmhd_MEdiag_N{}.eps".format(N[0]))

print('Generated Energy Figures.')


### Anisotropy measure

fig_KE, axesKE = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#fig_KE.suptitle(r'Kinetic Anisotropy: $\langle u_1^2\rangle /\langle u_1^2+u_2^2\rangle$')
fig_ME, axesME = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#fig_ME.suptitle(r'Magnetic Anisotropy: $\langle b_1^2\rangle /\langle b_1^2+b_2^2\rangle$')
#fig_q, axesq = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#fig_q.suptitle(r'Vorticity Anisotropy: $\langle q_x^2\rangle /\langle q_x^2+q_y^2\rangle$')
#fig_j, axesj = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#fig_j.suptitle(r'Current Anisotropy: $\langle j_x^2\rangle /\langle j_x^2+j_y^2\rangle$')

for index in range(len(mag_values)):
    axesKE.plot(tt, u12_vector[index][:-1]/(u12_vector[index][:-1]+u22_vector[index][:-1]),\
        lw=lw, linestyle = lines_index[index], color=color_index[index],label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
    axesME.plot(tt, b12_vector[index][:-1]/(b12_vector[index][:-1]+b22_vector[index][:-1]),\
        lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
    #axesq.plot(tt, qx2_vector[index][:-1]/(qx2_vector[index][:-1]+qy2_vector[index][:-1]),\
    #    lw=lw, color=color_index[index], label=r'$M^2 = $'+fmt(M2_vector[index]))
    #axesj.plot(tt, jx2_vector[index][:-1]/(jx2_vector[index][:-1]+jy2_vector[index][:-1]),\
    #    lw=lw, color=color_index[index], label=r'$M^2 = $'+fmt(M2_vector[index]))

for axes in [axesKE,axesME]: #,axesq,axesj]:    
    axes.grid(True)
    axes.set_xlabel("t")
    axes.set_ylim([0,1])

axesKE.set_ylabel(r"Anisotropy(u)")
axesME.set_ylabel(r"Anisotropy(b)")
#axesq.set_ylabel(r"Anisotropy(q)")
#axesj.set_ylabel(r"Anisotropy(j)")

for fig in [fig_KE, fig_ME]: #, fig_q, fig_j]:
    if lg: 
        fig.legend( loc='lower center',fancybox=True, shadow=True, ncol=4)
        fig.subplots_adjust(bottom=bspace)
    fig.tight_layout()

if save_format == 'png' or save_format == 'both':
    fig_KE.savefig("qgmhd_KEanisotropy_N{}.png".format(N[0]),dpi=dpi)
    fig_ME.savefig("qgmhd_MEanisotropy_N{}.png".format(N[0]),dpi=dpi)
    #fig_q.savefig("qgmhd_qanisotropy_N{}.png".format(N[0]))
    #fig_j.savefig("qgmhd_janisotropy_N{}.png".format(N[0]))
if save_format == 'eps' or save_format == 'both':
    fig_KE.savefig("qgmhd_KEanisotropy_N{}.eps".format(N[0]))
    fig_ME.savefig("qgmhd_MEanisotropy_N{}.eps".format(N[0]))
    #fig_q.savefig("qgmhd_qanisotropy_N{}.eps".format(N[0]))
    #fig_j.savefig("qgmhd_janisotropy_N{}.eps".format(N[0]))

print('Generated Anisotropy Figures.')


### Lorentz Force measure

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#fig.suptitle(r'Lorentz Force (normalized by max): $M^2 \sqrt{\langle(b\cdot\nabla j)^2\rangle}$')

for index in range(len(mag_values)):
    axes.plot(tt, lor_vector[index][:-1]/np.amax(lor_vector[index]), lw=lw, \
        color=color_index[index], linestyle = lines_index[index],label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))

axes.grid(True)
axes.set_xlabel("t");
axes.set_ylabel(r"$F_L/max(F_L)$")
if lg: 
    fig.legend( loc='lower center',fancybox=True, shadow=True, ncol=4)
    fig.subplots_adjust(bottom=bspace)
fig.tight_layout()

if save_format == 'png' or save_format == 'both':
    fig.savefig("qgmhd_lorentz_single_N{}.png".format(N[0]),dpi=dpi)
if save_format == 'eps' or save_format == 'both':
    fig.savefig("qgmhd_lorentz_single_N{}.eps".format(N[0]))

print('Generated Lorentz Figures.')


### Conserved quantitites

#figH, axesH = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#figH.suptitle(r'Relative Error in $H = \iint({\bf u}\cdot M{\bf b} )dxdy$')
#figA, axesA = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#figA.suptitle(r'Relative Error in $\Phi = \iint\frac{1}{2}(MA)^2dxdy$')

#for index in range(len(mag_values)):
#
#    Hplot = (Hel_vector[index][:-1]-Hel_vector[index][0])/Hel_vector[index][0]
#    Aplot = (A2_vector[index][:-1]-A2_vector[index][0])/A2_vector[index][0]
#    axesH.plot(tt, Hplot, lw=lw, color=color_index[index], label=r"$M^2 = $"+fmt(M2_vector[index]))
#    axesA.plot(tt, Aplot, lw=lw, color=color_index[index], label=r"$M^2 = $"+fmt(M2_vector[index]))

#for axes in [axesH,axesA]:    
#    axes.grid(True)
#    axes.set_xlabel("t");

#axesH.set_ylabel(r"$e_r(H)$")
#axesH.set_ylabel(r"$e_r(A^2)$")

#for fig in [figH, figA]:
#    fig.legend( loc='lower center',fancybox=True, shadow=True, ncol=4)
#    fig.tight_layout()
#    fig.subplots_adjust(bottom=bspace)

#if save_format == 'png' or save_format == 'both':
#    figH.savefig("qgmhd_crosshel_N{}.png".format(N[0]))
#    figA.savefig("qgmhd_meanA2_N{}.png".format(N[0]))
#if save_format == 'eps' or save_format == 'both':
#    figH.savefig("qgmhd_crosshel_N{}.eps".format(N[0]))
#    figA.savefig("qgmhd_meanA2_N{}.eps".format(N[0]))

#print('Generated Conservation Figures.')


### L_u and L_b

figU, axesU = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#figU.suptitle(r'Kinetic Microscale: $L_u = \langle {\bf u\cdot u}\rangle^{1/2}/\langle q^2\rangle^{1/2}$')
figB, axesB = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#figB.suptitle(r'Magnetic Microscale: $L_b = \langle {\bf b\cdot b}\rangle^{1/2}/\langle j^2\rangle^{1/2}$')

for index in range(len(mag_values)):
    axesU.semilogy(tt, Lu_vector[index][:-1], lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
    axesB.semilogy(tt, Lb_vector[index][:-1], lw=lw, linestyle = lines_index[index], color=color_index[index], label=r'$M = $'+fmt(np.sqrt(M2_vector[index])))
for axes in [axesU,axesB]:    
    axes.grid(True)
    axes.set_xlabel("t")
    axes.set_ylim([0.025,3])

axesU.set_ylabel(r"$L_u$")
axesB.set_ylabel(r"$L_b$")

for fig in [figU, figB]:
    if lg: 
        fig.legend( loc='lower center',fancybox=True, shadow=True, ncol=4)
        fig.subplots_adjust(bottom=bspace)
    fig.tight_layout()

if save_format == 'png' or save_format == 'both':
    figU.savefig("qgmhd_Lu_N{}.png".format(N[0]),dpi=dpi)
    figB.savefig("qgmhd_Lb_N{}.png".format(N[0]),dpi=dpi)
if save_format == 'eps' or save_format == 'both':
    figU.savefig("qgmhd_Lu_N{}.eps".format(N[0]))
    figB.savefig("qgmhd_Lb_N{}.eps".format(N[0]))

print('Generated Microscale Figures.')


### Growth rate

I1 = np.array( np.where(((tt > t_bound[0]) & (tt < t_bound[1]))) )
I1 = I1.ravel()
p1_array = []; pp1_array = []; p2_array = []; pp2_array = [];

figq, axesq = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#figq.suptitle(r'Norm of the q perturbation $||q-\bar{q}||$')
figa, axesa = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#figa.suptitle(r'Norm of the A perturbation $||A-\bar{A}||$')

if len(I1) > 2:
    for index in range(len(mag_values)):
        p1 = np.polyfit(tt[I1], np.log(qgrow_vector[index][I1]), 1)
        pp1 = np.exp(np.polyval(p1,(tt)))
        p2 = np.polyfit(tt[I1], np.log(Agrow_vector[index][I1]), 1)
        pp2 = np.exp(np.polyval(p2,(tt)))
        print('q and A growth rates for '+str(M2_vector[index])+' are ', p1[0], p2[0])
        axesq.semilogy(tt, qgrow_vector[index][:-1],  lw=lw, linestyle = lines_index[index], color=color_index[index], \
            label=r'$M = $'+fmt(np.sqrt(M2_vector[index]))+": Slope = {p:2.2f}".format(p=p1[0]))
        axesa.semilogy(tt, Agrow_vector[index][:-1],  lw=lw, linestyle = lines_index[index], color=color_index[index], \
            label=r'$M = $'+fmt(np.sqrt(M2_vector[index]))+": Slope = {p:2.2f}".format(p=p2[0]))
        #axesq.semilogy(tt[I1], pp1[I1], '--m',  lw=3)
        #axesa.semilogy(tt[I1], pp2[I1], '--c',  lw=3)
else:
    raise(IndexError("Growth rate interpolation range too small."))

for axes in [axesq,axesa]:    
    axes.grid(True)
    axes.set_xlabel("t");

axesq.set_ylabel(r'$||q-\bar{q}||$')
axesa.set_ylabel(r'$||A-\bar{A}||$')

for fig in [figq, figa]:
    if lg: 
        fig.legend( loc='lower center',fancybox=True, shadow=True, ncol=4)
        fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()

if save_format == 'png' or save_format == 'both':
    figq.savefig("qgmhd_qgrow_N{}.png".format(N[0]),dpi=dpi)
    figa.savefig("qgmhd_agrow_N{}.png".format(N[0]),dpi=dpi)
if save_format == 'eps' or save_format == 'both':
    figq.savefig("qgmhd_qgrow_N{}.eps".format(N[0]))
    figa.savefig("qgmhd_agrow_N{}.eps".format(N[0]))


print('Generated Growth Rates Figure.')

