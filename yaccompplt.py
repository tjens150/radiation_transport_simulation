from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle

blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)


# def Ledd(M):
#     return 1.26*10**31*M

# def Y():
#     return (2/(1+xe))*tau/4*(1-5/2*tau)**(1/3.)*mp/me

# def Ts():
#     return Y()*(1+Y()/0.27)**(-1/3.)
    
# def L():
#     return Ts()*calJ(Ts())Mdot()**2

def spacialint(g,dR,dt):
    diff=np.repeat(dR[:,np.newaxis],dt.shape[0],axis=1)
    return (g*diff).sum(axis=0)

def yacine(ainj,astart,Erat):
    aarr=np.linspace(afinal,ainj,10000)
    expo=np.exp(c/Erat*sigT*nh0/(H0*np.sqrt(omegaM))*2/3*(aarr**(-3/2.)-ainj**(-3/2)))
    return expo*(ainj**4/aarr**7)*nh0*c*sigT/Erat,aarr
    

#CONSTANTS
G=6.67408*10**(-11)*1.989*10**(30) #m^3 M_sol^(-1) s^(-2)

me=.511*10**6 #eV
mp=938.27**6
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
#CONSTANTS


fol='/Users/Acolyte/NYU/Research/rad_trans/full5000Photons/'
zread1=[1300,1250,1150,1000,900,800,700,600,500,400]
logr=np.linspace(np.log(0.1),np.log(500),nrbins)
rbins=np.insert(np.exp(logr),0,0)
rbins=np.append(rbins,100000)
astart=1/(max(zread1)+1)
afinal=1/51.
astep=0.1
alogbins=np.arange(np.log(astart),np.log(afinal),astep)
a32=np.exp(3/2.*alogbins)
dt=2/3.*(a32[1:]-a32[:-1])/(H0*np.sqrt(omegaM))
dR=4*pi/3*(rbins[1:]**3-rbins[:-1]**3)
denom=np.einsum('i,j->ij',dR,dt)




fil='comp/ardep_comp_v2.pkl'

with open(fol+fil,"rb") as f:
    ardep=pickle.load(f)
    adist=pickle.load(f)
    rdist=pickle.load(f)
    nphotbin=pickle.load(f)
    totnum=pickle.load(f)
    astat=pickle.load(f)
    Estat=pickle.load(f)


zbins=1/np.exp(alogbins)-1
#Erat=1/0.1

for nam in ardep:
    print(nam)
    fig,ax=plt.subplots(1,1)
    E=Estat[nam][0]
    nphot=totnum[nam]
    a=astat[nam][0]
    zwh=np.where(zbins <= 1/a-1)[0]
    if zwh[0]-1 > -1:
        zwhmod=np.insert(zwh,0,zwh[0]-1)
        tmpdenom=np.insert(denom,zwh[0],dR*2/3.*(a32[zwh[0]]-a**(3/2.))/(H0*np.sqrt(omegaM)),axis=1)
        tmpdenom=np.delete(tmpdenom,zwh[0]-1,axis=1)
    else:
        tmpdenom=denom
        #for nam in ['z1266_E0.11']:
    rhodep=ardep[nam][:,:,0]/tmpdenom
    G=rhodep/(nphot*E)
    Gt=spacialint(G,dR,dt)
    if Estat[nam][0]/me <0.2:
        Erat=1/0.069
    else:
        Erat=1/0.1067
    Gty,aarr=yacine(a,astart,Erat)
    zplot=(zbins[1:]+zbins[:-1])/2
    zwh=np.where(zplot <= 1/a-1)[0]
    zwh=np.append(zwh,zwh[0]-1)
    amean=(adist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    zmean=amean
    # zmean=1/(amean)-1
    ax.plot(1/aarr-1,Gty,lw=2,color=blue,zorder=0)
    ax.scatter(zmean[zwh],Gt[zwh],marker='*',s=15,color=orange,zorder=1)
    err=np.sqrt(adist[nam][:,:,1].sum(axis=0)/nphotbin[nam].sum(axis=0)-amean*amean)
    #err=np.sqrt(err)/(amean*amean)
    ax.errorbar(zmean[zwh],Gt[zwh],xerr=err[zwh],ecolor=orange,zorder=1,fmt=None)
    #ax.set_aspect(10)
    ax.set_xlabel('z')
    ax.set_ylabel(r'G (s$^{-1}$)')
    ax.set_title(r'$E_i=$%s $(m_e)$, $z_i=$%s, $N_\gamma=$%s' % (int(E/me*1000)/1000., int(1/a-1), nphot))
    #ax.set_yscale('log')
    plt.savefig(fol+'/figures/Gcomp/Gdep_comp'+nam+'.pdf')
    plt.close(fig)
#plt.show()



