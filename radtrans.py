from __future__ import division
import numpy as np
from math import pi
import pickle as pickle
from mpi4py import MPI
from scipy.stats import binned_statistic_2d
from _pulla_cy import pulla
import pdb


#CONSTANTS all in EV, meters, seconds
me=.511*10**6
zfinal=50.
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
h=0.67
aeq=4.15e-5/(omegaM*h**2)



#COEFFICIENCTS FOR EFFICIENCY
ptmpco=c*nh0/(np.sqrt(omegaM)*H0) #used for the C-code
Lstepco=c*2/(H0*np.sqrt(omegaM))   #used to compute comoving length

def diffthet(cthet,Ei): #unnormalized cosine of angle scattered PDF as a function of initial energy
    return (-1 + cthet**2 + 1/(Ei*(1/Ei - (-1 + cthet)/me)) + Ei*(1/Ei - (-1 + cthet)/me))/(Ei**2*(1/Ei - (-1 + cthet)/me)**2)

def sigthet(Ei): #diffthet integrated over angle (or Ef)
    return (2*me*((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/Ei**3

def pdfthet(cthet,Ei): #normalized pdf to sample the angle scattered
    return diffthet(cthet,Ei)/sigthet(Ei)

def maxPT(Ei): #max of the pdf of angle scattered is forward for any Ei.
    return pdfthet(1, Ei)

def rej(a,b,f,g,*args): 
    px=0.
    y=1.
    thismax=g(*args)
    while y >=px:  
        x=np.random.uniform(a,b)
        y=np.random.uniform(0,g(*args))
        px=f(x,*args)
    return x

#the rotation matrix R that rotates (sin(th) cos(ph), sin(th) sin(ph), cos(th)) into (0, 0, 1)
def rotmat(thet,phi): 
    return [[np.cos(phi)*np.cos(thet),np.sin(phi)*np.cos(thet),-np.sin(thet)],
            [-np.sin(thet),np.cos(phi),0],
            [np.sin(thet)*np.cos(phi),np.sin(phi)*np.sin(thet),np.cos(thet)]]

def rej(a,b,f,g,*args): 
    px=0.
    y=1.
    thismax=g(*args)
    while y >=px:  
        x=np.random.uniform(a,b)
        y=np.random.uniform(0,g(*args))
        px=f(x,*args)
    return x

def get_new_coords(Ei, ai, x,y,z, amax):
    seed=np.random.randint(0,2147483647) #seed for the C code
    
    #Cythonized Python code (C), evolves through time checking if the photon has scattered
    #outputs the time it scattered, and what the energy was after the redshifting.
    a, E=pulla(ai,Ei,amax,ptmpco,aeq,me,sigT,seed)
    chi =Lstepco*((aeq+a)**(1/2.)-(aeq+ai)**(1/2.)) #Comoving distance between ai and a
    newr = [x, y, z+chi]
    
    # draw theta, phi and rotate
    theta = np.arccos(rej(-1.,1.,pdfthet,maxPT,E))
    phi   = np.random.uniform(0,2*pi)
    R     = rotmat(theta,phi)
    newr  = np.dot(R, newr)
    Ef    = 1/(2/me*np.sin(theta/2.)**2+1/E)  # energy after scattering given theta   
    return Ef, a, E-Ef, newr[0], newr[1], newr[2]

# For a photon starting at the spatial origin at ai with energy Ei, tabulate energy loss as a function of redshift and distance
# Up to a maximum scale factor amax
def tabulate_one_photon(Ei, ai, amax):
    
    rtab    = [0]
    atab    = [ai]
    DEtab   = [0]

    a, E, x, y, z = ai, Ei, 0, 0, 0
    
    while a < amax:
        E, a, DE, x, y, z = get_new_coords(E, a, x, y, z,amax)
        rtab.append(np.sqrt(x**2 + y**2 + z**2))
        atab.append(a)
        DEtab.append(DE)
    return np.asarray(rtab)[1:-1]/mpc, np.asarray(atab)[1:-1], np.asarray(DEtab)[1:-1]   # getting rid of origin and psuedo-scatter after amax


#Short hand to bin a quantity and sum them and their squares via scipy
def binstats(r,a,quant,thisbin):
    thissum=np.array(binned_statistic_2d(r,a,quant,statistic='sum', bins=thisbin)[0])
    thissq=np.array(binned_statistic_2d(r,a,quant*quant,statistic='sum', bins=thisbin)[0])
    return np.stack((thissum,thissq),axis=2)

#CDF for a flat energy spectrum with an Emax and Emin
def flatEspec(samp,Eparams):
    Emax=Eparams[0]
    Emin=Eparams[1]
    return Emax**samp/Emin**(samp-1)*me

#Dirac delta energy spectrum, trivially returns Eret.
def diracEspec(samp,Eret):
    return Eret*me

# Computes histograms for delta E, a, z, and number of scatterings for Nphot photons injected at ai,
# using any energy spcetrum function's CDF. Also returns total energy injected from Nphot photons
def bin_photons(ai, Espec, amax, Nphot, rbins, abins, Eparams, rank=None, npstate=None):
    #Initialize arrays for binning, both the sum of quantity and sum of the square for variance
    result=np.zeros([len(rbins)-1,len(abins)-1,2]) #delta E
    photcount=np.zeros([len(rbins)-1,len(abins)-1]) #number of scatterings in each bin
    asum=np.zeros([len(rbins)-1,len(abins)-1,2]) #statistics for a
    zsum=np.zeros([len(rbins)-1,len(abins)-1,2]) #statistics for z
    rsum=np.zeros([len(rbins)-1,len(abins)-1,2]) #statistics for r
    Etot=0. #Total energy injected from Nphot sampled from Espec
    for i in range(Nphot):
        Ei=Espec(np.random.uniform(),Eparams)
        Etot+=Ei
        newr, aph, dEph=tabulate_one_photon(Ei,ai,amax)
        if len(newr) < 1:
            continue

        result += binstats(newr,aph,dEph,[rbins,abins])
        photcount += binstats(newr,aph,np.ones_like(newr,dtype='int'),[rbins,abins])[:,:,0]
        asum += binstats(newr,aph,aph,[rbins,abins])
        zsum += binstats(newr,aph,1/aph-1,[rbins,abins])
        rsum += binstats(newr,aph,newr,[rbins,abins])
        
        if (rank == 0) & (not i % 1000):
            print(i,'of ',N)

    return result, photcount, asum, zsum, rsum, Etot

if __name__ == '__main__':

    Ntot=4000000
    ai=1/1501. #before this, energy injection is negligible 
    amax=1/51. #beyond this reionization becomes relevant
    Emax=.447 #cut off for free-free (flat) energy spectrum of a PBH after recomb. (units of me)
    Emin=0.045 #Lower energy bound from which to sample, below this (2*pi*fine structure), photon wavelength is larger than bohr radius. (units of me)
    Edirac=5. #if using dirac injection (units of me)

    #2d bin initialization
    astep=0.1
    abins=np.exp(np.arange(np.log(ai),np.log(amax),astep))

    nrbins=70
    logr=np.linspace(np.log(1),np.log(600),nrbins)
    rbins=np.insert(np.exp(logr),0,0)
    rbins=np.append(rbins,100000)

    comm = MPI.COMM_WORLD #fork CPUs

    rank = comm.Get_rank() #get unique identifier for each CPU
    size = comm.Get_size() #number of CPUs
    npstate=np.random.RandomState() #reinitialize a random state for each CPU
    #divide for each CPU
    N=int(Ntot / size)
    if not rank:
        N+=(Ntot-N*size)

    result,photcount,asum,zsum,rsum,Etot = bin_photons(ai, diracEspec, amax, N, rbins, abins, Edirac, rank=rank, npstate=npstate)
    
    #Gather the arrays from all the CPUs
    result=comm.gather(result,root=0)
    photcount=comm.gather(photcount,root=0)
    asum=comm.gather(asum,root=0)
    zsum=comm.gather(zsum,root=0)
    rsum=comm.gather(rsum,root=0)
    Etot=comm.gather(Etot,root=0)
    print(result)
    if not rank:
        #sum them together
        mresult=np.zeros_like(result[0])
        mphotcount=np.zeros_like(photcount[0])
        masum=np.zeros_like(asum[0])
        mzsum=np.zeros_like(zsum[0])
        mrsum=np.zeros_like(rsum[0])
        mEtot=0.
        for ss in range(len(result)):
            mphotcount+=photcount[ss]
            mresult+=result[ss]
            masum+=asum[ss]
            mzsum+=zsum[ss]
            mrsum+=rsum[ss]
            mEtot+=Etot[ss]

        #save the binned statistics
        tit='z'+str(int(1/ai-1))+'_'+'E_dirac'+str(Edirac)+'_N'+str(Ntot)+'_binned_v2'
        print(tit)
        fil=tit+'.pkl'
        with open('./pickle/'+fil, "wb") as f:
            pickle.dump(mresult,f)
            pickle.dump(mphotcount,f)
            pickle.dump(masum,f)
            pickle.dump(mzsum,f)
            pickle.dump(mrsum,f)
            pickle.dump(mEtot,f)
