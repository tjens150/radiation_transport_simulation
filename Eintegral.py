from __future__ import division
import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
from math import pi
import pickle as pickle
import colormaps as cmaps
from scipy import optimize
import time

plt.register_cmap(name='viridis', cmap=cmaps.viridis)
cmapV=plt.cm.get_cmap('viridis')

blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps*a**(3/2.)/(sqrtomegaM*H0))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def deps(Ei,a):
    return 1/(H0*np.sqrt(omegaM)*a**(3/2.))*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def pXODEnewt(Ei,a, X, h):
    return 1+h+h*deps(Ei,a)

def cross(Ei):
    return (3/8.)*sigT*me/(Ei*Ei)*(2*(Ei**3 + 9*Ei*Ei*me+8*Ei*me*me+2*me**3)/((2*Ei+me)*(2*Ei+me))+(Ei-2*me-2*me*me/Ei)*np.log(1+2*Ei/me))

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args)#,disp=False)

def meanLsq(xode,aarr):
    iso=np.zeros(xode.size)
    crossiter=cross(xode)
    ai=aarr[0]
    def tmpf(aint):
        return c*aint**(3/2.)/(sqrtomegaM*H0*nh0)
    fint=tmpf(aarr)/crossiter
    for i in range(iso.size):
        finttmp=fint[0:i+1]
        atmp=aarr[0:i+1]
        iso[i]=((finttmp[:-1]+finttmp[1:])*(atmp[1:]-atmp[:-1])).sum()/2
    # pdb.set_trace()
    return iso

def find_nearest(array, value):
    array = np.asarray(array)
    array[~np.isfinite(array)]=0.
    idx = (np.abs(array - value)).argmin()
    return idx


#CONSTANTS everything in meters
G=6.67408*10**(-11)*1.989*10**(30) #m^3 M_sol^(-1) s^(-2)
me=.511*10**6 #eV
mp=938.27**6
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
sqrtomegaM=np.sqrt(omegaM)
#CONSTANTS

zinj=1200
ainj=[1/(zinj+1.)]
afinal=1/51.
NE=1000
Na=10000


Earr=np.linspace(10*me,0.01*me,NE)

# comm = MPI.COMM_WORLD

# rank = comm.Get_rank()
# size = comm.Get_size()

# divy=int(NE/size)
# strt=rank*divy
# if rank == size-1:
#     divy=NE-divy*(size-1)

# Esep=np.zeros([divy,Na], dtype=float)
# L2sep=np.zeros([divy,Na], dtype=float)

fol='/Users/Acolyte/NYU/Research/rad_trans/Espace/'

print('Reading in data...')
with open(fol+'Espace_log_z%s.pkl' % zinj, "rb") as f:
    Etot=pickle.load(f)
    L2tot=pickle.load(f)
    Earr=pickle.load(f)
    adisc=pickle.load(f)

dE=Earr[:-1]-Earr[1:]
Liso=np.sqrt(L2tot)
rres=1000
rmax=300 #mpc
#dr=rmax/rres
rlinarr=np.linspace(np.log10(1),np.log10(rmax),rres)
rlinarr=np.insert(10**(rlinarr),0,0)
# rarr=np.repeat(rarr[:,:,np.newaxis],Liso[0,:].size,axis=2)
# gaus=4*pi*rarr**3*np.exp(-3*(rarr)**2/(2*Liso**2))/(3/2.*np.sqrt(3/2.)/(pi**(3/2.)*Liso**3))
# print('Integrating over E and normalizing...')
# unEgaus=(gaus[:,1:,:]+gaus[:,:-1,:]).sum(axis=1)*dE/2
# Enorm=(unEgaus[1:,:]+unEgaus[:-1,:]).sum(axis=0)*dr/2


stepres=300
zplts=-np.arange(-(1/adisc[0]-1)+50,-(1/adisc[-1]-1),stepres)
Eplts=np.array([10,5,1,0.1,0.01])*me
rgord=np.linspace(0,1,len(zplts)+1)
rgordpad=1/(len(zplts)*5/4.)
#rgba=cmapV(rgord)
fig,ax=plt.subplots(1,1)

for ee in range(len(Eplts)):
    indxE=find_nearest(Earr,Eplts[ee])
    for cc in range(len(zplts)):
        LEiso=Liso[indxE:,:]
        Egord=np.linspace(rgord[cc],rgord[cc]+rgordpad,len(Eplts))
        rgba=cmapV(Egord)
        indx=find_nearest(1/adisc-1,zplts[cc])
        rarr=np.repeat(rlinarr[:,np.newaxis],LEiso[:,0].size,axis=1)
        Ltmp=np.repeat((LEiso[:,indx])[np.newaxis,:],rlinarr.size,axis=0)
        gaus=np.exp(-3*(rarr)**2/(2*Ltmp**2))#/(3/2.*np.sqrt(3/2.)/(pi**(3/2.)*Ltmp**3))
        gaus[np.isclose(Ltmp, 0)]=0
        dEarr=np.repeat(dE[np.newaxis,indxE:],gaus.shape[0],axis=0)
        unEgaus=((gaus[:,1:]+gaus[:,:-1])*dEarr).sum(axis=1)/2
        lnrgauss=4*pi*rlinarr**2*unEgaus
        Enorm=((lnrgauss[1:]+lnrgauss[:-1])*(rlinarr[1:]-rlinarr[:-1])).sum(axis=0)/2
        ax.axvspan(LEiso[-1,indx],LEiso[0,indx] ,zorder=0,color=rgba[ee],alpha=0.1)
        sigint=(lnrgauss*rlinarr**2)/Enorm
        sigfit=((sigint[1:]+sigint[:-1])*(rlinarr[1:]-rlinarr[:-1])).sum(axis=0)/2
        # meanL=np.mean(LEiso[:,indx])
        meanL=np.sqrt(sigfit)
        testgaus=4*pi*rlinarr**3*np.exp(-3*(rlinarr)**2/(2*meanL**2))*(3/2.*np.sqrt(3/2.)/(pi**(3/2.)*meanL**3))
#        pdb.set_trace()
        ax.plot(rlinarr,testgaus,lw=2,color=rgba[ee], linestyle='-.', zorder=0,alpha=0.4)
        # testgaus=4*pi*rlinarr**3*np.exp(-3*(rlinarr)**2/(2*LEiso[0,indx]**2))*(3/2.*np.sqrt(3/2.)/(pi**(3/2.)*LEiso[0,indx]**3))
        # ax.plot(rlinarr,testgaus,lw=2,color=rgba[ee], linestyle=':', zorder=0,alpha=0.5)
        # ax.axvline(x=LEiso[0,indx],lw=2,linestyle=':',zorder=0,color=rgba[ee],alpha=0.85)
        ax.axvline(x=meanL,lw=2,linestyle=':',zorder=0,color=rgba[ee],alpha=0.4)
        # ax.axvline(x=LEiso[-1,indx],lw=2,linestyle=':',zorder=0,color=rgba[ee],alpha=0.85)
        if ee == 0:
            ax.plot(rlinarr,rlinarr*lnrgauss/Enorm,lw=2,color=rgba[ee], linestyle='-', zorder=cc,label='%s' % int(1/adisc[indx]-1))
        else:
            ax.plot(rlinarr,rlinarr*lnrgauss/Enorm,lw=2,color=rgba[ee], linestyle='-', zorder=cc)

ax.legend(fontsize=14,loc='upper left',ncol=1,labelspacing=0.2,handlelength=0.75)
ax.set_xscale('log')
ax.set_xlabel(r'$r$ (Mpc)',fontsize=20)
ax.set_ylabel(r'$4\pi r^3$G',fontsize=20)
ax.set_title(r'$z_{\rm inj}=$%s, $E_{\rm cut}\in %s \,\,(m_e)$' %(zinj,list(Eplts/me)))
#ax.set_aspect(.5)
plt.savefig(fol+'figures/Eint_z%s_cut_insp_log.pdf' % zinj)             
plt.show()
# for a in ainj:
#     adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
#     dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
#     for Ei in range(divy):
#         Esep[Ei,0]=Earr[strt+Ei]
#         for ev in (np.arange(Na-1)+1):
#             Esep[Ei,ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Esep[Ei,ev-1],dlna)
#         L2sep[Ei,:]=meanLsq(Esep[Ei,:],adisc)/mpc**2

#         if rank ==0:
#             #if not divy % 10:
#             print('%s of %s' % (Ei*size,NE))


# Ecom=comm.gather(Esep,root=0)
# L2com=comm.gather(L2sep,root=0)
# if rank == 0:
#     Etot=Ecom[0]
#     L2tot=L2com[0]
#     for i in np.arange(size-1)+1:
#         Etot=np.concatenate((Etot,Ecom[i]),axis=0)
#         L2tot=np.concatenate((L2tot,L2com[i]),axis=0)
