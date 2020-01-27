from __future__ import division
import numpy as np
from math import pi
import pickle as pickle
import pdb
import matplotlib.pyplot as plt

#CONSTANTS all in EV, meters, seconds
me=.511*10**6 #electron mass
c=299792458.0 #speed of light
mpc=3.086*10**22 #Mpc in m

#Cosmological parameters
h=0.67
YHe=0.24 #fraction of Helium from BBN
H0=100.*h*1000./mpc # Hubble constant in seconds
T0=2.73 #CMB temp today
Nnueff=3.046 #dof of Neutrinos
omegaM=0.1409/h**2 #matter density
omegaR=T0**4*4.48162687719e-7*(1+0.227107318*Nnueff)/h**2 #relativistic density
omegaK=0. #Curvature density
omegaDE=1-omegaM-omegaR-omegaK #Dark Energy density
aeq=4.15e-5/(omegaM*h**2) #matter radiation equaltiy
omegab=0.02226/h**2 #baryon density
nh0=(1-YHe)*11.3*omegab*h**2 #density of hydrogen today

def HubbleRate(a): #Compute Hubble
    return H0*np.sqrt(omegaM/a**3+omegaK/a**2+omegaDE+omegaR/a**4)

def spacialint(g,dR,dt): #function to integrate Green's function over space
    diff=np.repeat(dR[:,np.newaxis],dt.shape[0],axis=1)
    return (g*diff).sum(axis=0)

def dtfunc(a1,a2): #Compute the time between two scale factors in a matter+rad dom
    return 2/(3*H0*np.sqrt(omegaM))*(a2*np.sqrt(a2+aeq)-a1*np.sqrt(a1+aeq)+2*aeq*(np.sqrt(a1+aeq)-np.sqrt(a2+aeq)))

def find_nearest(array, value): #Returns index of the entry in array closest to value
    array = np.asarray(array)
    array[~np.isfinite(array)]=0.
    idx = (np.abs(array - value)).argmin()
    return idx


#Class that initializes with the [r,a] bins used from radtrans.py when binning
#Generates the Green's function, and its spatially averaged temporal part
#Can also plot the spatial Green's function
class spatial_greens: 


    rbins=None #radial deposition bins
    abins=None #temporal deposition bins
    G=None #2d in r-dep and a-dep green's function for a given injection time
    Gnorm=None #Spatially integrated G, 1d in a-dep, overall normalization

    #read in data from radtrans.py
    nphotbin=None #number of scatters in each bin
    adist=None #sum of a and a^2 of scattering in each bin for statistics
    zdist=None #sum of z and z^2 of scattering in each bin for statistics
    rdist=None #sum of r and r^2 of scattering in each bin for statistics

    def __init__(self, rbins, abins): #bins used to generate histogram in radtrans.py
        self.rbins=rbins
        self.abins=abins
        
    #With a given initial ai of injection
    #Loads in the binned statistics pickle file generated from radtrans.py
    #and stores the Green's function, and its temporal part.
    def gen_greens(self,ai,pikfil=None): 
        with open(pikfil,"rb") as f:
            ardep=pickle.load(f) #sum of dE in each bin
            self.nphotbin=pickle.load(f)
            self.adist=pickle.load(f)    
            self.zdist=pickle.load(f)
            self.rdist=pickle.load(f)
            Etot=pickle.load(f) #total energy injected into simulation
        dR=4*pi/3*(self.rbins[1:]**3-self.rbins[:-1]**3) #volume of each bin
        dt=dtfunc(self.abins[:-1],self.abins[1:]) #time of each bin
        denom=dR[:,np.newaxis]*dt[np.newaxis,:]
        self.G=ardep[:,:,0]/(denom*Etot*HubbleRate(ai))
        self.Gnorm=spacialint(self.G,dR,dt)

    
    #Given the injection information i.e., time of injection, energy, and total photons
    #plots the normalized spatial Green's functions at zplts,
    #then optionally saves the plot in savfol.
    def plot_spatial(self,ai, Eparam, Ntot, zplts, savfol=None):

        #Colors for plot
        import colormaps as cmaps
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        cmapV=plt.cm.get_cmap('viridis')
        rgord=np.linspace(0,1,len(zplts))
        rgba=cmapV(rgord)

        fig, ax=plt.subplots(1,1)

        G, Gnorm = self.G, self.Gnorm
        zmean=(self.zdist[:,:,0].sum(axis=0))/self.nphotbin.sum(axis=0) #mean z for each temporal bin
        rmean=(self.rdist[:,:,0])/self.nphotbin #mean r of scatter population in each bin
        zbins=1/self.abins-1
        for cc in range(len(zplts)):
            indx=find_nearest(zmean,zplts[cc]) #find the bin closest to desired z-plot
            cut=(self.nphotbin[:,indx]>100) #arbitrary cut off for high-number statistics
            ax.plot(rmean[cut,indx],4*pi*rmean[cut,indx]**3*G[cut,indx]/Gnorm[indx],lw=2,color=rgba[cc], zorder=cc,label=r'(%s$\leq z\leq$ %s)' % (int(zbins[indx]),int(zbins[indx+1])))

        ll=ax.legend(fontsize=10,loc='upper left',ncol=1,labelspacing=0.2,handlelength=0.75)
        ll.set_zorder(10000)
        ax.set_xlabel(r'$r$ (Mpc)',fontsize=20)
        ax.set_ylabel(r'$4\pi r^3$G',fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(r'$E_{\rm inj}=$%s $(m_e)$, $z_{\rm inj}=$%s, $N_{\rm tot}=$%s' % (int(Eparam*1000)/1000., int(1/ai-1), Ntot), fontsize=18)
        ax.set_xscale('log')
        fig.tight_layout()
        if savfol is None:
            plt.show()
        else:
            plt.savefig(savfol+'z%s_E%s_N%s' % (int(1/ai-1),int(Eparam),int(Ntot)))
            plt.close(fig)
