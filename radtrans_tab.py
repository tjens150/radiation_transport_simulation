from __future__ import division
import numpy as np
from math import pi
import pickle as pickle
from mpi4py import MPI #Parallel module--This doesn't actually work for Jupyter
from scipy.stats import binned_statistic
from scipy import interpolate
#from _pulla_cy import pulla
import time
import pdb


#CONSTANTS all in EV, meters, seconds
me=.511*10**6 #electron mass
c=299792458.0 #speed of light
mpc=3.086*10**22 #Mpc in m
zfinal=50. #end of simulation
sigT=6.652459*10**(-29.) #thomson cross section

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


#Background ionization history from HyRec                                                            
ldf=np.loadtxt('./bg_xe.dat')
zint=ldf[:,0] #loads in z arr
xeint=ldf[zint >= 20.,1] #reionization becomes relevant after z=50
zint=zint[zint >=20.]
np.where(xeint >1, xeint, 1.) #code breaks with xe>1, ignore Helium, it is recombined near H recomb
xe=interpolate.interp1d(1/(zint+1),xeint)


#COEFFICIENCTS FOR EFFICIENCY
ptmpco=c*nh0/(np.sqrt(omegaM)*H0) #used for the C-code
Lstepco=c*2/(H0*np.sqrt(omegaM))   #used to compute comoving length
compco=3*sigT*me/8.
Heco=YHe/(4.*(1.-YHe))

def diffthet(cthet,Ei): #unnormalized cosine of angle scattered PDF as a function of initial energy
    return (-1 + cthet**2 + 1/(Ei*(1/Ei - (-1 + cthet)/me)) + Ei*(1/Ei - (-1 + cthet)/me))/(Ei**2*(1/Ei - (-1 + cthet)/me)**2)

def sigthet(Ei): #diffthet integrated over angle (or Ef)
    return (2*me*((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/Ei**3

def pdfthet(cthet,Ei): #normalized pdf to sample the angle scattered
    return diffthet(cthet,Ei)/sigthet(Ei)

def maxPT(Ei): #max of the pdf of angle scattered is forward for any Ei.
    return pdfthet(1, Ei)

def eta_H(E):
    return 1./np.sqrt(E/13.6-1) #13.6 for ionizing Hydrogen

def sig_H(E):
    eta=eta_H(E)
    # 2**9*M_PI**2*rnot**2/(3*alph**3)*Eth**4=1.178E-15 m^2*eV
    return 1.178e-15*(1/E)**4*np.exp(-4*eta*np.arctan(1/eta))/(1-np.exp(-2*np.pi*eta))

def sig_He(E,sigH):
    return -12.*sigH+5.1e-24*(250./E)**3.3

def maxPT(Ei): #max of the pdf of angle scattered is forward for any Ei.
    return pdfthet(1, Ei)

#Acceptance Rejection method of sampling from the pdf f, in a domain of [a,b]
#g is max(f) in the domain
#This randomly pulls x from the domain, and then pulls y randomly from [0,g],
# if  y < f(x) then accept, else pull another x and start over.
def rej(a,b,f,g,Earr,randarr): 
    px=0.
    y=1.
    crit=np.ones_like(Earr,dtype=bool)
    x=np.zeros_like(Earr)
    #pdb.set_trace()
    while np.any(crit):#y >=px
        thisrand=randarr[:Earr.size][crit]
        thisE=Earr[crit]
        x[crit]=thisrand*(b-a)+a
        randarr=cycle_prob(randarr)
        thisrand=randarr[:Earr.size][crit]
        y=thisrand*g(thisE)
        px=f(x[crit],thisE)
        thiscrit=crit[crit]
        thiscrit[y<px]=False
        (crit[crit])=thiscrit
        randarr=cycle_prob(randarr)
    return x

#the rotation matrix R that rotates (sin(th) cos(ph), sin(th) sin(ph), cos(th)) into (0, 0, 1)
# def rotmat(thet,phi): 
#     return [[np.cos(phi)*np.cos(thet),np.sin(phi)*np.cos(thet),-np.sin(thet)],
#             [-np.sin(phi),np.cos(phi),0],
#             [np.sin(thet)*np.cos(phi),np.sin(phi)*np.sin(thet),np.cos(thet)]]

def rotmat(thet,phi): 
    return np.asarray(((np.cos(phi)*np.cos(thet),np.sin(phi)*np.cos(thet),-np.sin(thet)),
            (-np.sin(phi),np.cos(phi),np.zeros_like(thet)),
            (np.sin(thet)*np.cos(phi),np.sin(phi)*np.sin(thet),np.cos(thet))))

def compt_scat(Eistep,randarr):     # draw theta, phi and dE 
    theta = np.arccos(rej(-1.,1.,pdfthet,maxPT,Eistep,randarr))
    phi   = randarr[-Eistep.size:]*2*pi
    Ef    = 1/(2/me*np.sin(theta/2.)**2+1/Eistep)  # energy after scattering given theta   
    return Eistep-Ef, theta, phi

def probcoeff(astep): #Rad+Mat dominated a-dependent prob coeff
    return ptmpco/np.sqrt(astep*astep*astep*(1+aeq/astep))

def probscat(Eistep): #prob of compton scattering
    return compco/(Eistep*Eistep)*(2.*(Eistep*Eistep*Eistep + 9.*Eistep*Eistep*me+8.*Eistep*me*me+2.*me**3)/((2.*Eistep+me)*(2.*Eistep+me))+(Eistep-2.*me-2.*me*me/Eistep)*np.log(1.+2.*Eistep/me))

def probHion(xestep,sigH): #prob of Hydrogen ionization
    return (1.-xestep)*sigH

def probHeion(Eistep,sigH): #prob of Helium ionization
    return Heco*sig_He(Eistep,sigH)

def cycle_prob(randarr):
    return np.roll(randarr,np.int(randarr[0]*randarr.size)) #cycle through random arr for psuedo-psuedo-random

def check_act(Eiarr,astep,dstep,not_term,randarr):
    Eistep=Eiarr[not_term]
    thisrand=randarr[not_term]
    retflag=np.zeros((3,not_term.size),dtype=bool)
    sigH=sig_H(Eistep)
    proba=probcoeff(astep) #a-dependent coefficient for probability
    scat=probscat(Eistep) #scattering probability
    He=probHeion(Eistep,sigH)
    dlna=dstep*np.min((1,1/(proba*scat.max()),1/(proba*He.max()))) #set the time-step
    coeff=dlna*proba
    retflag[0,not_term]=(thisrand<coeff*probHion(xe(astep),sigH))
    thisrand=cycle_prob(thisrand)
    retflag[1,not_term]=(thisrand<coeff*He)  #dummy Helium ionization flag
    thisrand=cycle_prob(thisrand)    
    retflag[2,not_term]=(thisrand < coeff*scat)
    return retflag, dlna

#Short hand to bin a quantity and sum them and their squares via scipy
def binstats(r,quant,thisbin):
    return np.asarray(binned_statistic(r,quant,statistic='sum', bins=thisbin)[0])


# Computes histograms for delta E, a, z, and number of scatterings for Nphot photons injected at ai,
# using any energy spcetrum function's CDF. Also returns total energy injected from Nphot photons
def bin_photons(ai, abinind, rarr, dEph, rbins, abins,result,photcount,asum):
    #Initialize arrays for binning, both the sum of quantity and sum of the square for variance
    result[:,abinind] += binstats(rarr,dEph,rbins)
    photcount[:,abinind] += binstats(rarr,np.ones_like(rarr,dtype='int'),rbins)
    asum[:,abinind] += binstats(rarr,np.full_like(rarr,ai),rbins)        
    return result, photcount, asum


def evolve_arr(Einit,ai,amax,dlnastep,rbins,abins,progint):
    astep=np.copy(ai)
    Eistep=np.copy(Einit)
    vec=np.zeros((3,Eistep.size))
    thisdE=np.zeros(Eistep.size)
    randarr=np.random.uniform(size=(Eistep.size))
    result=np.zeros([len(rbins)-1,len(abins)-1]) #delta E
    photcount=np.zeros([len(rbins)-1,len(abins)-1]) #number of scatterings in each bin
    asum=np.zeros([len(rbins)-1,len(abins)-1]) #statistics for a
    thisN=1.
    progarr=np.exp(np.linspace(np.log(ai),np.log(amax),progint)) #array of checkpoints
    not_term=np.ones_like(Eistep,dtype=bool) #boolean array denoting not terminated photons
    tt=time.time()
    ttot=time.time()
    tarr=np.zeros_like(progarr)
    tcount=0
    abinindprev=0
    while astep<amax:
        abinind=np.argmax((astep-abins[1:])<0.)
#         if abinindprev != abinind:
#             abinindprev=np.copy(abinind)
#             thisdE=np.zeros(Eistep.size)
        
        flagarr,dlna=check_act(Eistep,astep,dlnastep,not_term,randarr) #retrieve truth array of each three processes and dlna
        ionbool=(flagarr[0,:]+flagarr[1,:]) #Ionization (H+He) boolean and not terminated
        not_term*=(~ionbool)
        if ~np.any(not_term):
            break
        if np.any(ionbool):  
            thisdE[flagarr[1,:]]=Eistep[flagarr[1,:]]-24.6 #compute dE Helium ionization
            thisdE[flagarr[0,:]]=Eistep[flagarr[0,:]]-13.6 #compute dE Hydrogen ionization
            Eistep[ionbool]= 0. #Deposit energy for H or He ionization
        
        comptbool=(~ionbool)*flagarr[2,:] #Compton scattering boolean (Ionization supersedes Compton)
        if np.any(comptbool):
            randarr=cycle_prob(randarr)
            dEcompt,theta,phi=compt_scat(Eistep[comptbool],randarr) #For compton scattered photons, compute dE and angle
            Eistep[comptbool]-=dEcompt #Deposit Compton energy
        
            thisdE[comptbool]=dEcompt #add dE from Compton
        #chi=Lstepco*((aeq+astep)**(1/2.)-(aeq+ai)**(1/2.))
        
        allbool=ionbool+comptbool
        if np.any(allbool):
            result,photcount,asum=bin_photons(astep, abinind, np.sqrt((vec[:,allbool]**2).sum(axis=0))/mpc,
                thisdE[allbool]/thisN, rbins, abins,result,photcount,asum)
        
        #prepare for next iteration
        vec[2,:]+=Lstepco*((aeq+astep*(1+dlna))**(1/2.)-(aeq+astep)**(1/2.))
        if np.any(comptbool):
            vec[:,comptbool]=np.einsum('ij...,j...->i...',rotmat(theta,phi),vec[:,comptbool])
        
        Eistep=Eistep/(1+dlna)
        astep=astep*(1+dlna)
        #print(1/astep-1)
        nalive=np.count_nonzero(not_term)
        #nterm=np.int(thisN/2-1)
        if nalive < np.floor(Einit.size/2):
            tmp=vec[:,~not_term]
            tmp[:,:nalive]=vec[:,not_term]
            vec[:,~not_term]=tmp
            
            tmp=thisdE[~not_term]
            tmp[:nalive]=thisdE[not_term]
            thisdE[~not_term]=tmp

            tmp=Eistep[~not_term]
            tmp[:nalive]=Eistep[not_term]
            Eistep[~not_term]=tmp
            
            tmp=not_term[~not_term]
            tmp[:nalive]=True
            not_term[~not_term]=tmp

            thisN*=2.

        if np.any(np.isclose(astep,progarr,atol=dlna/2.*astep,rtol=0.)):
            tarr[tcount]=time.time()-tt
            print('Current z=%0.2f, with zmax=%0.2f. %s alive photons. Elapsed Time= %0.2f s (dlna= %.2E)' % (1/astep-1,1/amax-1,nalive,tarr[tcount],dlna))
            tt=time.time()
            tcount+=1
        randarr=cycle_prob(randarr)
    print('Total time: %s' % (time.time()-ttot))
    return result,photcount,asum,tarr

#CDF for a flat energy spectrum with an Emax and Emin
def flatEspec(samp,Eparams):
    Emax=Eparams[0]
    Emin=Eparams[1]
    return Emax**samp/Emin**(samp-1)*me

#Dirac delta energy spectrum, trivially returns Eret.
def diracEspec(samp,Eret):
    return np.full_like(samp,Eret*me)

def init_Earr(N,Espec,Eparams):
    Earr=Espec(np.random.uniform(size=N),Eparams)
    Etot=Earr.sum()
    return Earr,Etot



if __name__ == '__main__':
    Ntot=1000000
    ai_bin=1/1501. #before this, energy injection is negligible                                  
    amax=1/51. #beyond this reionization becomes relevant                                        
    #cut off for free-free (flat) energy spectrum of a PBH after recomb. (units of me)          
    Emax=.447 #collisional ionization case                                                       
    Emax_PI=20.1 #photoionized case                                                              
    Emin=0.045 #Lower energy bound from which to sample, below this (2*pi*fine structure), photon wavelength is larger than bohr radius. (units of me)                                           
    Edirac=5 #if using dirac injection (units of me)                                             

    #2d bin initialization                                                                       
    astep=0.01
    abins=(np.arange(np.log(ai_bin),np.log(amax),astep))
    tp=abins+astep/4.
    tm=abins-astep/4.
    abins=np.empty((tp.size+tm.size),dtype=tp.dtype)
    abins[0::2]=np.exp(tm)
    abins[1::2]=np.exp(tp)
    nrbins=140
    logr=np.linspace(np.log(1),np.log(1000),nrbins)
    rbins=np.insert(np.exp(logr),0,0)
    rbins=np.append(rbins,100000)

    nEbins=70
    logE=np.linspace(np.log(1),np.log(me*Edirac),nEbins)
    Ebins=np.insert(np.exp(logE),0,0)

    fol='./pickle/photion/tab/'

    afinal=1/101.                                                                                
    aliststep=0.025                                                                              
    ailist=np.exp(np.arange(np.log(ai_bin),np.log(afinal),aliststep))                            

    comm = MPI.COMM_WORLD #fork CPUs                                                             

    rank = comm.Get_rank() #get unique identifier for each CPU                                   
    size = comm.Get_size() #number of CPUs                                                       
    np.random.seed() #reinitialize a random state for each CPU                                   
    #divide for each CPU
    ailist=ailist[rank::size]

    dlnastep=0.01
    progint=100.
    ailist=[ailist[0]]
    #    ailist=[ai_bin]                                                                         
    for ai in ailist:                  
        Einit,Etot=init_Earr(Ntot,flatEspec,[Emax_PI,Emin])
        result,photcount,asum,tarr=evolve_arr(Einit,ai,abins.max(),dlnastep,rbins,abins,progint)
        tit='z'+str(int(1/ai-1))+'_'+'E_flat'+str(Emax_PI)+'_N'+str(Ntot)+'_dlna'+str(dlnastep)+'_tab_PI'
        print(tit)
        fil=tit+'.pkl'
        with open(fol+fil, "wb") as f:
            pickle.dump(result,f)
            pickle.dump(photcount,f)
            pickle.dump(asum,f)
            pickle.dump(Etot,f)
