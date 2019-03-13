from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle
import colormaps as cmaps
#from scipy.linalg import solve_triangular
#from scipy.sparse import diags
from scipy import optimize
import time

plt.register_cmap(name='viridis', cmap=cmaps.viridis)


blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)

def spacialint(g,dR,dt):
    diff=np.repeat(dR[:,np.newaxis],dt.shape[0],axis=1)
    return (g*diff).sum(axis=0)

def yacine(ainj,astart,Erat):
    aarr=np.linspace(afinal,ainj,10000)
    expo=np.exp(c/Erat*sigT*nh0/(H0*np.sqrt(omegaM))*2/3*(aarr**(-3/2.)-ainj**(-3/2)))
    return expo*(ainj**4/aarr**7)*nh0*c*sigT/Erat,aarr

def diffcross(Ei, Ef):
    return (3/8.)*sigT*me/(Ei*Ei)*(Ef/Ei+Ei/Ef-1+(1+me/Ei-me/Ef)**2)

def eps(Ei,a):
    return a**(-3.)*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def deps(Ei,a):
    return 1/(H0*sqrtomegaM*a**(3/2.))*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps*a**(3/2.)/(sqrtomegaM*H0))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def pXODEnewt(Ei,a, X, h):
    return 1+h+h*deps(Ei,a)

def XODEtest(Ei,a,X,h):
    tmpeps=-(Ei+nh0*a**(-3.)*c*0.1*sigT*Ei*(a**(3/2.)/(sqrtomegaM*H0)))
    return Ei-h*tmpeps-X

def pXODEtest(Ei,a,X,h):
    tmpeps=-(1+nh0*a**(-3.)*c*0.1*sigT*(a**(3/2.)/(sqrtomegaM*H0)))
    return 1-h*tmpeps

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args)

def ODEG(adisc,XODEval,E0,dlna):
    #exp=deps(XODEval,adisc)/adisc
    expt=(nh0*c*0.1*sigT/(sqrtomegaM*H0))*2/3.*(adisc[0]**(-3/2.)-adisc**(-3/2.))
    # exp=(nh0/adisc**3*c*0.1*sigT*(adisc**3/2/(sqrtomegaM*H0)))/adisc
    nsize=adisc.size
    expint=np.zeros(nsize)
    tmpeps=nh0*adisc**(-3.)*c*0.1*sigT*XODEval
    # for i in np.arange(nsize-1)+1:
    #     expint[i]=((exp[:(i-nsize)]+exp[1:(i+1)])*dlna*adisc[1:(i+1)]).sum()/2
    # pdb.set_trace()
    # return np.exp(expint)*(adisc[0]/adisc)**2*eps(XODEval,adisc)/XODEval
    expint=expt
    return np.exp(expint)*(adisc[0]/adisc)**2*tmpeps/XODEval
def newG(adisc,XODEval,E0):
    return (adisc[0]/adisc)**3*eps(XODEval,adisc)/E0



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
sqrtomegaM=np.sqrt(omegaM)
#CONSTANTS



fol='/Users/Acolyte/NYU/Research/rad_trans/full5000Photons/'
nrbins=50
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
# amean=1.





fil='comp/ardep_comp_v2.pkl'

with open(fol+fil,"rb") as f:
    ardep=pickle.load(f)
    adist=pickle.load(f)    
    zdist=pickle.load(f)
    rdist=pickle.load(f)
    Edist=pickle.load(f)
    nphotbin=pickle.load(f)
    totnum=pickle.load(f)
    astat=pickle.load(f)
    Estat=pickle.load(f)


zbins=1/np.exp(alogbins)-1
#Erat=1/0.1

for nam in ardep:
    print(nam)
    # if nam != 'z1250_E5':
    #     continue
    fig,ax=plt.subplots(1,1)
    E=Estat[nam][0]
    nphot=totnum[nam]
    a=astat[nam][0]
    zwh=np.where(zbins <= 1/a-1)[0]
    zmean=(zdist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    amean=(adist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    ameantest=(np.exp(alogbins[1:])+np.exp(alogbins[:1]))/2
    denom=np.einsum('i,j->ij',dR,dt*(amean/a)**3)
    Na=100000
    dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
        
    adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
    print('Beginning PDE solver...')
    ttot=time.time()
    PDErho=np.zeros(Na)
    PDErho[0]=np.sqrt(a)*nphot/(H0*np.sqrt(omegaM))
    Xstep=np.zeros(Na)
    Xstep[0]=E
    Xstept=np.copy(Xstep)
    for ev in (np.arange(Na-1)+1):
        Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)
        #Xstept[ev]=imptstep(XODEtest,pXODEtest,adisc[ev],Xstep[ev-1],dlna)
        #PDErho[ev]=PDErho[ev-1]/(1-dlna*deps(Xstep[ev],adisc[ev]))
        # if not ev % 10000:
        #     print(ev)
            
    print('Finished PDE solver.')
    print('PDE took: %s seconds' % (time.time()-ttot))
    #PDErho=PDErho*eps(Xstep,adisc)/(adisc*adisc)
    #PDErho=dlnE*(uin[1:-1,:]*epsarr).sum(axis=0)/(adisc*adisc*adisc)
    #GODE=ODEG(adisc,Xstept,E,dlna)
    NEWODE=newG(adisc,Xstep,E)
    if zwh[0]-1 > -1:
        # zwhmod=np.insert(zwh,0,zwh[0]-1)
        tmpdenom=np.insert(denom,zwh[0],dR*(amean[zwh[0]-1]/a)**3*2/3.*(a32[zwh[0]]-a**(3/2.))/(H0*np.sqrt(omegaM)),axis=1)
        tmpdenom=np.delete(tmpdenom,zwh[0]-1,axis=1)
    else:
        tmpdenom=denom
        #for nam in ['z1266_E0.11']:
    rhodep=ardep[nam][:,:,0]/tmpdenom
    G=rhodep/(nphot*E)
    Gt=spacialint(G,dR,dt)
    if Estat[nam][0]/me <0.2:
        # Erat=1/0.069
        Erat=1/0.1
        ticks=[np.log10(0.1),np.log10(0.08),np.log10(0.05),np.log10(0.01),np.log10(0.005)]
        ticknames=[0.1,0.08,0.05,0.01,0.005]
    else:
        # Erat=1/0.1067
        Erat=1/0.1
        ticks=[np.log10(5),np.log10(3),np.log10(1),np.log10(0.1),np.log10(0.05),np.log10(0.01)]
        ticknames=[5,3,1,0.1,0.05,0.01]

    Gty,aarr=yacine(a,astart,Erat)
    
    zplot=(zbins[1:]+zbins[:-1])/2
    zwh=np.where(zplot <= 1/a-1)[0]
    zwh=np.append(zwh,zwh[0]-1)
    Emean=np.log10((Edist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)/me)
    # zmean=1/(amean)-1
    ax.plot(1/aarr-1,Gty,lw=2.5,color=blue,zorder=0,label='AK 2017 Approx.')
    ax.plot(1/adisc-1,NEWODE,lw=2.5,color=red,zorder=1,label='Exact')
    sc=ax.scatter(zmean[zwh],Gt[zwh],c=Emean[zwh],marker='o',s=12,zorder=3,cmap='viridis',edgecolors='none')
    cbar=plt.colorbar(sc)
    cbar.set_label(r'$E_{{\rm mean}}$/$m_e$',fontsize=18)
    err=np.sqrt(zdist[nam][:,:,1].sum(axis=0)/nphotbin[nam].sum(axis=0)-zmean*zmean)
    #err=np.sqrt(err)/(amean*amean)
    _,__,errorlines=ax.errorbar(zmean[zwh],Gt[zwh],xerr=err[zwh],c=Emean[zwh],zorder=2,fmt=None, cerror='black',elinewidth=2)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticknames)
    cbar.ax.tick_params(labelsize=16)
    errorlines[0].set_color(cbar.to_rgba(Emean[zwh]))
    #errorlines[1].set_color(cbar.to_rgba(Emean[zwh]))
    #ax.set_aspect(10)
    ax.legend(fontsize=18,loc='lower right')
    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel(r'G (s$^{-1}$)',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(r'$E_{\rm inj}=$%s $(m_e)$, $z_{\rm inj}=$%s, $N_\gamma=$%s' % (int(E/me*1000)/1000., int(1/a-1), nphot), fontsize=18)
    plt.savefig(fol+'/figures/GcompPDE/'+nam+'.pdf')
    ax.set_yscale('log')
    #ax.plot(1/adisc-1,GODE,lw=2,color=green,zorder=1)
    plt.savefig(fol+'/figures/GcompPDE/log/'+nam+'.pdf')

    plt.close(fig)
#plt.show()



