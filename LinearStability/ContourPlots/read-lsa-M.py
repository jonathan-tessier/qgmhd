# LSAQGMHD: reae lsa-M.py output 

# BY: Jonathan Tessier, June 2025

import numpy as np
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import sys
import h5py

# Output parameters for pickup files in HDF5
filename = "output.h5"                        # output filename
file = h5py.File(filename, mode="r")          # output file

L    = file["L"][...]; #print(repr(L))
N    = file["N"][...]; #print(repr(N))
Ne   = int(file["Ne"][...]); #print(repr(Ne))
F    = file["F"][...]
MM2  = file["MM2"][...]; NM = len(MM2)
kk   = file["kk"][:]; Nk = len(kk)
Re   = file["Re"][...]
Rm   = file["Rm"][...]
beta = file["beta"][...]
Lj   = file["Lj"][...]
Uj   = file["Uj"][...]

grow = file["grow"][:] # (Ne,Nk,NM)
freq = file["freq"][:] # (Ne,Nk,NM)
p_modes = file["p_mode"][:] # (N+1,Ne,Nk,NM)
a_modes = file["a_mode"][:] # (N+1,Ne,Nk,NM)

print("File "+filename+" opened")
print("Computational Parameters:")
print("Domain Length (L) = "+str(L))
print("Domain Num Points = "+str(N))
print("Domain Resolution = "+str(L/N))
print("")
print("Physical Parameters")
print("F    = "+str(F))
print("M2   = "+str(MM2))
print("beta = "+str(beta))
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

evn_grow = np.zeros([Nk,NM]) 
odd_grow = np.zeros([Nk,NM])

if True:
    for ii in range(Ne):
        for ik in range(Nk):
            for iM in range(NM):
                parity = abs(0.5*(np.sign(p_modes[int(N/2)-1,ii,ik,iM].real) - \
                np.sign(p_modes[int(N/2)+1,ii,ik,iM].real)))
                print("growth rate: "+str(grow[ii,ik,iM]))
                print("parity: "+str(parity))
                if parity == 0 and evn_grow[ik,iM] == 0:
                    evn_grow[ik,iM] = grow[ii,ik,iM]
                elif parity == 1 and odd_grow[ik,iM] == 0:
                    odd_grow[ik,iM] = grow[ii,ik,iM]
                else: 
                    print('weaker growth rate')

    plt.figure(figsize=(8,4))

    kplot,p_plot = np.meshgrid(kk,MM2)
    plt.subplot(121); plt.title("Even Mode for F = "+str(F))
    plt.contourf(kplot,p_plot,np.transpose(evn_grow[:,:]),cmap='Greys',levels=16) 
    plt.colorbar()
    plt.xlabel("k")
    plt.ylabel(r"$M^2$")

    plt.subplot(122); plt.title("Odd Mode for F = "+str(F))
    plt.contourf(kplot,p_plot,np.transpose(odd_grow[:,:]),cmap='Greys',levels=16) 
    plt.colorbar()
    plt.xlabel("k")
    plt.ylabel(r"$M^2$")

    plt.tight_layout()
    plt.savefig("QGMHD_Even_Odd.png") 

if True:

    # pick M value for strutures..
    Mind = 0

    plt.figure(figsize=(8,6))
    plt.clf()

    # Find max growth rates
    Imax = np.zeros((Ne,1), dtype=int)
    for ii in range(Ne):
        Imax[ii] = np.argmax(grow[ii,:,Mind])
        print("Max growth for curve", ii+1, "is", grow[ii,Imax[ii],Mind])
        print("Frequency for curve", ii+1, "is", freq[ii,Imax[ii],Mind])
        #plt.plot(kk[Imax[ii]],grow[ii,Imax[ii]],'o',markersize=10) # mark max growth

    plt.plot(kk,grow[0,:,Mind],'.k', )
    plt.plot(kk,grow[1,:,Mind],'.k', )
    plt.plot(kk,grow[2,:,Mind],'.k', )
    plt.plot(kk,grow[3,:,Mind],'.k', )
    plt.grid(True)
    plt.title("Growth Rates (N = "+str(N)+"): "+r"$M^2$ = "+'{:.2f}'.format(MM2[Mind])+", F = "+'{:.2f}'.format(F))
    #plt.legend(loc='best')
    plt.grid('on')
    plt.xlabel('wavenumber')
    plt.ylabel('growth')
    plt.tight_layout()
    plt.savefig("QGMHD_growth_bickley_spectral_"+str(N)+".png") 
    #plt.show()

if True:
    for which_mode in range(4):
        # Plot the four most unstable modes
        plt.figure(figsize=(4,4))
        plt.clf()
        # order modes by parity...
        dataset = np.transpose(grow[which_mode,:,:])
        kplot,p_plot = np.meshgrid(kk,MM2)
        plt.contourf(kplot,p_plot,dataset,cmap='Greys',levels=16) 
        plt.title("F = "+str(F))
        plt.xlabel("k")
        plt.ylabel(r"$M^2$")
        #plt.clim([0,1])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("QGMHD_growth_Mcontour_Mode"+str(which_mode+1)+".png") 

if True:
    plt.figure(figsize=(12,12))
    plt.clf()

    for ii in range(Ne):

        plt.subplot(Ne,2,1+2*ii)
        plt.plot(y,p_modes[:,ii,Imax[ii],Mind].real,'.-b')
        plt.plot(y,p_modes[:,ii,Imax[ii],Mind].imag,'.-r')

        parity = abs(0.5*(np.sign(p_modes[int(N/2)-1,ii,Imax[ii],Mind].real) - \
                np.sign(p_modes[int(N/2)+1,ii,Imax[ii],Mind].real)))
        print(parity)

        plt.title('Psi mode '+str(ii+1)+':  growthrate = {:.2e}, midpt = {:.3e}'.format(grow[ii,Imax[ii],Mind][0],\
                p_modes[int(N/2),ii,Imax[ii],Mind][0].real))
        plt.subplot(Ne,2,2+2*ii)
        plt.plot(y,a_modes[:,ii,Imax[ii],Mind].real,'.-b')
        plt.plot(y,a_modes[:,ii,Imax[ii],Mind].imag,'.-r')
        plt.title('A mode '+str(ii+1)+':  frequency = {:.2e}'.format(freq[ii,Imax[ii],Mind][0]))
        plt.tight_layout()   
    plt.savefig("QGMHD_eigen_bickley_spectral_"+str(N)+".png")     

    plt.figure(figsize=(8,8))
    
    for ii in range(Ne):
   
        plt.clf()
 
        Lx = 2*np.pi/kk[Imax[ii]]
        print("2x Domain width: ",2*Lx)
        x = np.linspace(0,8*np.pi,N+1)
        xx,yy = np.meshgrid(x,y)
    
        psi_mode2d = (np.tile(p_modes[:,ii,Imax[ii],Mind],(1,N+1))*np.exp(2*np.pi*1j*xx/Lx)).real
        mag_mode2d = (np.tile(a_modes[:,ii,Imax[ii],Mind],(1,N+1))*np.exp(2*np.pi*1j*xx/Lx)).real

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
  



