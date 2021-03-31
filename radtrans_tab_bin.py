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

# #fraction of electron energy that goes into ICS <10.2 photons (energy sink)
# with open('./gam102.pkl',"rb") as f:
#     Earr102=pickle.load(f)
#     aarr102=np.flip(pickle.load(f))
#     gam102=np.flip(pickle.load(f),axis=1)

# frac102=interpolate.RectBivariateSpline(Earr102,aarr102,gam102) #interpolation must be strictly ascending, hence the flips. Also to remind grid=False for pulling values


with open('./gam_all.pkl',"rb") as f:
    Earr102=pickle.load(f)
    aarr102=np.flip(pickle.load(f))
    rat_ion=np.flip(pickle.load(f),axis=1)
    rat_exc=np.flip(pickle.load(f),axis=1)
    rat_heat=np.flip(pickle.load(f),axis=1)

frac_ion=interpolate.RectBivariateSpline(Earr102,aarr102,rat_ion) 
frac_exc=interpolate.RectBivariateSpline(Earr102,aarr102,rat_exc) 
frac_heat=interpolate.RectBivariateSpline(Earr102,aarr102,rat_heat) 
#interpolation must be strictly ascending, hence the flips. Also to remind grid=False for pulling values


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
    count=0
    while np.any(crit):#y >=px
        thisrand=randarr[:Earr.size][crit]
        thisE=Earr[crit]
        x[crit]=thisrand*(b-a)+a
        randarr=cycle_prob(randarr, fullrand=False)
        thisrand=randarr[-Earr.size:][crit]
        y=thisrand*g(thisE)
        px=f(x[crit],thisE)
        thiscrit=crit[crit]
        thiscrit[y<px]=False
        (crit[crit])=thiscrit
        randarr=cycle_prob(randarr, fullrand=False)
        count+=1
        if (count == 10000):
            print('Warning, got stuck in rej, resampling random')
            randarr=np.random.uniform(size=randarr.size)
            count=0
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

def cycle_prob(randarr,fullrand=True):
    if fullrand: return np.random.uniform(size=randarr.size)
    return np.roll(randarr,1) #np.roll(randarr,3) #cycle through random arr for psuedo-psuedo-random

def check_act(Eiarr,astep,dstep,minprob,not_term,randarr,photoio_flag=True):
    Eistep=Eiarr[not_term]
    thisrand=randarr[not_term]
    retflag=np.zeros((3,not_term.size),dtype=bool)
    sigH=sig_H(Eistep)
    proba=probcoeff(astep) #a-dependent coefficient for probability
    scat=probscat(Eistep) #scattering probability
    if photoio_flag: #Photoionization
        He=probHeion(Eistep,sigH)
        dlna=np.min((dstep,minprob/(proba*scat.max()),minprob/(proba*He.max()))) #set the time-step
        coeff=dlna*proba
        retflag[0,not_term]=(thisrand<coeff*probHion(xe(astep),sigH))
        thisrand=cycle_prob(thisrand)
        retflag[1,not_term]=(thisrand<coeff*He)  #dummy Helium ionization flag
        thisrand=cycle_prob(thisrand)
    else: 
        dlna=np.min((dstep,minprob/(proba*scat.max()))) #set the time-step
        coeff=dlna*proba
    retflag[2,not_term]=(thisrand < coeff*scat)
    return retflag, dlna

#Short hand to bin a quantity and sum them and their squares via scipy
def binstats(r,quant,thisbin):
    return np.asarray(binned_statistic(r,quant,statistic='sum', bins=thisbin)[0])


# Computes histograms for delta E, a, z, and number of scatterings for Nphot photons injected at ai,
# using any energy spcetrum function's CDF. Also returns total energy injected from Nphot photons
def bin_photons(ai, abinind, rarr, dEph, rbins, abins,result,photcount,asum,splitbinflag,splitfrac,thisN):
    #Initialize arrays for binning, both the sum of quantity and sum of the square for variance
    if splitbinflag: #if our step enters into a new bin, must split into both bins
        for i in range(3): #in order ion, exc, heat
            this_Efrac=frac_dep(ai,dEph,i)
            result[:,abinind,i] += binstats(rarr,this_Efrac*splitfrac,rbins)
            result[:,abinind-1,i] += binstats(rarr,this_Efrac*(1-splitfrac),rbins)
        photcount[:,abinind] += binstats(rarr,np.ones_like(rarr)*splitfrac,rbins)
        photcount[:,abinind-1] += binstats(rarr,np.ones_like(rarr)*(1-splitfrac),rbins)
    else:
        for i in range(3): #in order ion, exc, heat
            this_Efrac=frac_dep(ai,dEph,i)
            result[:,abinind,i] += binstats(rarr,this_Efrac,rbins)
        photcount[:,abinind] += binstats(rarr,np.ones_like(rarr),rbins)
    asum[:,abinind] += binstats(rarr,np.full_like(rarr,ai),rbins)  #for the time being splitting these other statistics is negligible  
    return result, photcount, asum

# def frac_dep(tmpastep,tmpdE,ics_flag=True):
#     # if len(tmpdE)<1: return np.asarray([])
#     if not ics_flag: return tmpdE
#     return tmpdE*(1-frac102(tmpdE,tmpastep,grid=False))

def frac_dep(tmpastep,tmpdE,i):
    if i == 0:
        return tmpdE*(frac_ion(tmpdE,tmpastep,grid=False))
    if i == 1:
        return tmpdE*(frac_exc(tmpdE,tmpastep,grid=False))
    if i == 2:
        return tmpdE*(frac_heat(tmpdE,tmpastep,grid=False))

def dtfunc(a1,a2): #Compute the time between two scale factors in a matter+rad dom
        return 2/(3*H0*np.sqrt(omegaM))*(a2*np.sqrt(a2+aeq)-a1*np.sqrt(a1+aeq)+2*aeq*(np.sqrt(a1+aeq)-np.sqrt(a2+aeq)))

def evolve_arr(Einit,ai,amax,dlnastep,minprob,rbins,abins,progint,ics_flag=True,photoio_flag=True):
    astep=np.copy(ai)
    Eistep=np.copy(Einit)
    vec=np.zeros((3,Eistep.size))
    thisdE=np.zeros(Eistep.size)
    randarr=np.random.uniform(size=(Eistep.size))
    result=np.zeros([len(rbins)-1,len(abins)-1,3]) #delta E in three channels, ion, exc, heat
    photcount=np.zeros([len(rbins)-1,len(abins)-1]) #number of scatterings in each bin
    asum=np.zeros([len(rbins)-1,len(abins)-1]) #statistics for a
    thisN=1.
    progarr=np.exp(np.linspace(np.log(ai),np.log(amax),progint)) #array of checkpoints
    not_term=np.ones_like(Eistep,dtype=bool) #boolean array denoting not terminated photons
    tt=time.time()
    ttot=time.time()
    tarr=np.zeros_like(progarr)
    tcount=0
    abinind=np.argmax((astep-abins[1:])<0.)
    abinindprev=np.copy(abinind)
    splitfrac=0 #fraction of energy that goes into bins if step overlaps two bins
    while astep<amax:
        splitbinflag=0 #flag indicating there is an overlap of astep of two bins
        abinind=np.argmax((astep-abins[1:])<0.)
        if abinindprev != abinind : 
            splitbinflag=1 
            splitfrac=dtfunc(abins[abinind],astep)/dtfunc(astep/(1+dlna),astep)
        abinindprev=abinind
#         if abinindprev != abinind:
#             abinindprev=np.copy(abinind)
#             thisdE=np.zeros(Eistep.size)
        
        flagarr,dlna=check_act(Eistep,astep,dlnastep,minprob,not_term,randarr,photoio_flag=photoio_flag) #retrieve truth array of each three processes and dlna
        ionbool=(flagarr[0,:]+flagarr[1,:]) #Ionization (H+He) boolean and not terminated
        not_term*=(~ionbool)
        if ~np.any(not_term):
            break
        if np.any(ionbool):
            thisdE[flagarr[1,:]]=frac_dep(astep,Eistep[flagarr[1,:]]-24.6,ics_flag=ics_flag) #compute dE Helium ionization
            thisdE[flagarr[0,:]]=frac_dep(astep,Eistep[flagarr[0,:]]-13.6,ics_flag=ics_flag) #compute dE Hydrogen ionization
            Eistep[ionbool]= 0. #Deposit energy for H or He ionization
        

        comptbool=not_term*flagarr[2,:] #Compton scattering boolean (Ionization supersedes Compton)
        if np.any(comptbool):
            randarr=cycle_prob(randarr)
            dEcompt,theta,phi=compt_scat(Eistep[comptbool],randarr) #For compton scattered photons, compute dE and angle
            Eistep[comptbool]-=dEcompt #Deposit Compton energy
        
            thisdE[comptbool]=frac_dep(astep,dEcompt,ics_flag=ics_flag) #add dE from Compton
        #chi=Lstepco*((aeq+astep)**(1/2.)-(aeq+ai)**(1/2.))
        
        allbool=ionbool+comptbool
        if np.any(allbool):
            result,photcount,asum=bin_photons(astep, abinind, np.sqrt((vec[:,allbool]**2).sum(axis=0))/mpc,
                                              thisdE[allbool], rbins, abins,result,photcount,asum,splitbinflag,splitfrac,thisN)
        
        #prepare for next iteration
        if np.any(comptbool):
            vec[:,comptbool]=np.einsum('ij...,j...->i...',rotmat(theta,phi),vec[:,comptbool])
        vec[2,:]+=Lstepco*((aeq+astep*(1+dlna))**(1/2.)-(aeq+astep)**(1/2.))
        
        Eistep=Eistep/(1+dlna)
        astep=astep*(1+dlna)
        #print(1/astep-1)
        nalive=np.count_nonzero(not_term)
        #nterm=np.int(thisN/2-1)
        if nalive < np.floor(Einit.size/2):
            tmp=vec[:,~not_term]
            tmp[:,:nalive]=vec[:,not_term]
            vec[:,~not_term]=tmp
            
            # tmp=thisdE[~not_term]
            # tmp[:nalive]=thisdE[not_term]
            # thisdE[~not_term]=tmp

            tmp=Eistep[~not_term]
            tmp[:nalive]=Eistep[not_term]
            Eistep[~not_term]=tmp
            
            tmp=not_term[~not_term]
            tmp[:nalive]=True
            not_term[~not_term]=tmp

            thisN*=2.
        
        thisdE=np.zeros(Eistep.size)
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
    ics_flag=False
    photoio_flag=False
    Ntot=1000000
    ai_bin=1/1501. #before this, energy injection is negligible                                  
    amax=1/51. #beyond this reionization becomes relevant                                        
    #cut off for free-free (flat) energy spectrum of a PBH after recomb. (units of me)          
    Emax=.447 #collisional ionization case                                                       
    Emax_PI=20.1 #photoionized case                                                              
    Emin=0.045 #Lower energy bound from which to sample, below this (2*pi*fine structure), photon wavelength is larger than bohr radius. (units of me)                                           
    Edirac=17.61609 #if using dirac injection (units of me)                                             
    mbh_T_arr=np.loadtxt('./MBH1_T.dat') #in units of me
    E_int=interpolate.interp1d(1/(mbh_T_arr[:,0]+1),mbh_T_arr[:,1],fill_value='extrapolate',bounds_error=False) # v_bc=0
    
    

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

    fol='./pickle/photion/tab/fixed_03_2021/channels/'
    
    #extend=np.flip(-np.arange(-np.log(abins[0])+astep/2.,-np.log(1/2001.)+astep/2.,astep/2.))
    #abins=np.insert(abins,0,np.exp(extend))
    afinal=1/101.                                                                                
    aliststep=0.025                                                                              
    ailist=np.exp(np.arange(np.log(ai_bin)-aliststep/2.,np.log(afinal)-aliststep/2.,aliststep/2.))         #DOUBLING RESOLUTION ATM
    #ailist=np.asarray([1/1301.,1/901.])
    #ailist=[1/1486.7734]#,1/958.7315]
    # ailist=ailist[-16::6]

    #afinal =1/1501.
    #aliststep=0.025/2.
    #newai=1/2001.
    #tt=np.exp(np.flip(-np.arange(-np.log(ailist[0])+aliststep/2.,-np.log(newai),aliststep/2.)))
    #ailist=tt
    
    # tt=np.asarray([1323,1290,1391,1426,1462,1110,1197,1167,932,1005,980,822],dtype=int)
    # cut=np.ones_like(ailist,dtype=bool)
    # for it in tt:
    #     cut=np.where((1/ailist-1).astype(int)==it,False,cut)
    # ailist=ailist[cut]
    comm = MPI.COMM_WORLD #fork CPUs                                                             

    rank = comm.Get_rank() #get unique identifier for each CPU                                   
    size = comm.Get_size() #number of CPUs                                                       
    np.random.seed() #reinitialize a random state for each CPU                                   
    #divide for each CPU
    #ailist=ailist[21::7]
    ailist=ailist[rank::size]
    minprob=0.01/2. #at max .25% chance of scattering
    dlnastep=0.005/2. #min step required from binning 2 times smaller than adep bin
    progint=100.
    for ai in ailist:
        #this_Emax=E_int(ai)
        this_Emax=Emax
        Einit,Etot=init_Earr(Ntot,flatEspec,[this_Emax,Emin])
        #Einit,Etot=init_Earr(Ntot,diracEspec,Edirac)
        result,photcount,asum,tarr=evolve_arr(Einit,ai,abins.max(),dlnastep,minprob,rbins,abins,progint,ics_flag=ics_flag,photoio_flag=photoio_flag)
        #tit='z'+str(int(1/ai-1))+'_'+'E_dirac'+str(Edirac)+'_N'+str(Ntot)+'_dlna'+str(dlnastep)+'_noICSnoPI'
        tit='z'+str(int(1/ai-1))+'_'+'E_flat'+str(this_Emax)+'_N'+str(Ntot)+'_dlna'+str(dlnastep)
        # tit='z'+str(int(1/ai-1))+'_'+'E_dirac'+str(Edirac)+'_N'+str(Ntot)+'_dlna'+str(dlnastep)+'_tab_frac_bin'
        print(tit)
        fil=tit+'.pkl'
        with open(fol+fil, "wb") as f:
            pickle.dump(result,f)
            pickle.dump(photcount,f)
            pickle.dump(asum,f)
            pickle.dump(Etot,f)
