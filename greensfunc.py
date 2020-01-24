from __future__ import division
import numpy as np
from math import pi
import pickle as pickle
import pdb
import matplotlib.pyplot as plt

#CONSTANTS all in EV, meters, seconds
me=.511*10**6
zfinal=50.
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
h=0.67
T0=2.73
Nnueff=3.046
omegaM=0.308
omegaR=T0**4*4.48162687719e-7*(1+0.227107318*Nnueff)/h**2
omegaK=0.
omegaDE=1-omegaM-omegaR-omegaK
aeq=4.15e-5/(omegaM*h**2)

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


    rbins=None
    abins=None
    G=None
    Gnorm=None
    nphotbin=None
    adist=None
    zdist=None
    rdist=None

    def __init__(self, rbins, abins): #bins used to generate histogram in radtrans.py
        self.rbins=rbins
        self.abins=abins
        
    #With a given initial ai of injection
    #Loads in the binned statistics pickle file generated from radtrans.py
    #and returns the Green's function, and its temporal part.
    def gen_greens(self,ai,pikfil=None): 
        with open(pikfil,"rb") as f:
            ardep=pickle.load(f)
            self.nphotbin=pickle.load(f)
            self.adist=pickle.load(f)    
            self.zdist=pickle.load(f)
            self.rdist=pickle.load(f)
            Etot=pickle.load(f)
        dR=4*pi/3*(self.rbins[1:]**3-self.rbins[:-1]**3)
        dt=dtfunc(self.abins[:-1],self.abins[1:])
        denom=dR[:,np.newaxis]*dt[np.newaxis,:]
        self.G=ardep[:,:,0]/(denom*Etot*HubbleRate(ai))
        self.Gnorm=spacialint(self.G,dR,dt)

    
    #Given the injection information i.e., time of injection, energy, and total photons
    #plots the normalized spatial Green's functions at zplts,
    #then optionally saves the plot in savfol.
    def plot_spatial(self,ai, Eparam, Ntot, zplts, savfol=None):
        import colormaps as cmaps
        plt.register_cmap(name='viridis', cmap=cmaps.viridis)
        cmapV=plt.cm.get_cmap('viridis')
        rgord=np.linspace(0,1,len(zplts))
        rgba=cmapV(rgord)
        fig, ax=plt.subplots(1,1)

        G, Gnorm = self.G, self.Gnorm
        zmean=(self.zdist[:,:,0].sum(axis=0))/self.nphotbin.sum(axis=0)
        rmean=(self.rdist[:,:,0])/self.nphotbin
        zbins=1/self.abins-1
        for cc in range(len(zplts)):
            indx=find_nearest(zmean,zplts[cc])
            cut=(self.nphotbin[:,indx]>100)
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
