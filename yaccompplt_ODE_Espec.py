from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle
#import colormaps as cmaps
#from scipy.linalg import solve_triangular
#from scipy.sparse import diags
from scipy import optimize
from scipy import interpolate
import time

#plt.register_cmap(name='viridis', cmap=plt.get_cmaps(viridis))


blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)

def HubbleRate(a):
    return H0*np.sqrt(omegaM/a**3+omegaK/a**2+omegaDE+omegaR/a**4)

def spacialint(g,dR,dt):
    diff=np.repeat(dR[:,np.newaxis],dt.shape[0],axis=1)
    return (g*diff).sum(axis=0)

def eps(Ei,a):
    return a**(-3.)*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def deps(Ei,a):
    return 1/(HubbleRate(a)*a**3)*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps/HubbleRate(a))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def pXODEnewt(Ei,a, X, h):
    return 1+h+h*deps(Ei,a)

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args)

def newG(adisc,XODEval,E0):
    return (adisc[0]/adisc)**3*eps(XODEval,adisc)/(E0*HubbleRate(adisc[0]))

def dtfunc(a1,a2):
    return 2/(3*H0*sqrtomegaM)*(a2*np.sqrt(a2+aeq)-a1*np.sqrt(a1+aeq)+2*aeq*(np.sqrt(a1+aeq)-np.sqrt(a2+aeq)))

# def dtfunc(a1,a2):
#     return 2/3.*(a2-a1)/(H0*np.sqrt(omegaM))

def find_nearest(array, value):
    array = np.asarray(array)
    array[~np.isfinite(array)]=0.
    idx = (np.abs(array - value)).argmin()
    return idx



#CONSTANTS
G=6.67408*10**(-11)*1.989*10**(30) #m^3 M_sol^(-1) s^(-2)
me=.511*10**6 #eV
mp=938.27**6
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
h=0.7
T0=2.73
Nnueff=3.046
omegaM=0.308
omegaR=T0**4*4.48162687719e-7*(1+0.227107318*Nnueff)/h**2
omegaK=0.
omegaDE=1-omegaM-omegaR-omegaK
sqrtomegaM=np.sqrt(omegaM)
aeq=4.15e-5/(omegaM*h**2)
#CONSTANTS



folsav='/home/data/tj796/Research/radtrans/figures/temporal_Espec/'
fol='/home/data/tj796/Research/radtrans/pickle/mbh100_rmin20_ainjstep0.05/'

nrbins=70
logr=np.linspace(np.log(20),np.log(600),nrbins)
rbins=np.insert(np.exp(logr),0,0)
rbins=np.append(rbins,100000)

astart=1/(1301.)
afinal=1/51.
zfinal=50.
astep=0.1
alogbins=np.arange(np.log(astart),np.log(afinal),astep)
#a32=np.exp(3/2.*alogbins)

#dt=2/3.*(a32[1:]-a32[:-1])/(H0*np.sqrt(omegaM))
alin=np.exp(alogbins)
dt=dtfunc(alin[:-1],alin[1:])
rmean=(rbins[1:]+rbins[:-1])/2.
dR=4*pi/3*(rbins[1:]**3-rbins[:-1]**3)#/rmean

#dRint=pi*(rbins[1:]**4-rbins[:-1]**4)
# amean=1.


mbh=100
T_file='PBH_MBH_%s_T_novbc.dat' % (mbh)
ldf=np.loadtxt(T_file)
Tfarr=ldf[:-1,-1]
zarr=ldf[:-1,0]
Tf=interpolate.interp1d(np.log(1/(1+zarr)),Tfarr,kind='cubic')

astartb=1/1301.
aliststep=0.05
alistiter=np.exp(np.arange(np.log(astartb),np.log(1/(zfinal+50)),aliststep))


fil='comp/ardep_comp_v2.pkl'

Emin=0.01
nEbins=20.




zbins=1/np.exp(alogbins)-1
#Erat=1/0.1
Ntot=100000

for astart in alistiter:
    tit='z'+str(int(1/astart-1))+'_'+'MBH'+str(mbh)+'_N'+str(Ntot)+'_binned.pkl'
    with open(fol+tit,"rb") as f:
        nphotbin=pickle.load(f)
        ardep=pickle.load(f)
        adist=pickle.load(f)    
        zdist=pickle.load(f)
        rdist=pickle.load(f)
        Edist=pickle.load(f)
    nam='z'+str(int(1/astart-1))+'_'+'MBH'+str(mbh)+'_N'+str(Ntot)
    Ecut=Tf(np.log(astart))
    Ebins=np.linspace(Emin,Ecut,nEbins)
    dE=(Ecut-Emin)/nEbins
    Etot=(Ebins*Ntot).sum()
    print(nam)
    # if nam != 'z1250_E5':
    #     continue
    

    nphot=Ntot*nEbins
    a=astart
    zwh=np.where(zbins <= 1/a-1)[0]
    zmean=(zdist[:,:,0].sum(axis=0))/nphotbin.sum(axis=0)
    amean=(adist[:,:,0].sum(axis=0))/nphotbin.sum(axis=0)
   # rmean=(rdist[:,:,0].sum(axis=0))/nphotbin.sum(axis=0)
    denom=np.einsum('i,j->ij',dR,dt*(amean/a)**3)
    Na=5000
    dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
        
    adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
    print('Beginning PDE solver...')
    ttot=time.time()
    NEWODE=np.zeros(Na)
    for ee in range(len(Ebins)):
        E=Ebins[ee]*me
        
        Xstep=np.zeros(Na)
        Xstep[0]=E
        Xstept=np.copy(Xstep)
        for ev in (np.arange(Na-1)+1):
            Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)
            
        if (ee == 0) or (ee == len(Ebins)-1):
            NEWODE+=newG(adisc,Xstep,E)*E/2.
        else:
            NEWODE+=newG(adisc,Xstep,E)*E
        NEWODE=dE*NEWODE/(Ecut*Ebins.sum()*me)
    print('Finished PDE solver.')
    print('%s PDEs took: %s seconds' % (len(Ebins),time.time()-ttot))
    if zwh[0]-1 > -1:
        # zwhmod=np.insert(zwh,0,zwh[0]-1)  amean[zwh[0]-1]
        tmpdenom=np.insert(denom,zwh[0],dR*(amean[zwh[0]-1]/a)**3*dtfunc(a,alin[zwh[0]]),axis=1)#2/3.*(a32[zwh[0]]-a**(3/2.))/(H0*np.sqrt(omegaM)),axis=1)

        tmpdenom=np.delete(tmpdenom,zwh[0]-1,axis=1)
    else:
        tmpdenom=denom
        #for nam in ['z1266_E0.11']:
    G=ardep[:,:,0]/tmpdenom
    Gt=spacialint(G,dR,dt)*dE/(Ecut*Etot*me)/(HubbleRate(a))#*8.831793880834816e-14#*7.8e-5
    #Gt=spacialint(G,pi*(rbins[1:]**4-rbins[:-1]**4),dt)*Ecut/Etot
    #pdb.set_trace()
    if Ecut/me <0.2:
        # Erat=1/0.069
        Erat=1/0.1
        ticks=[np.log10(0.1),np.log10(0.08),np.log10(0.05),np.log10(0.01),np.log10(0.005)]
        ticknames=[0.1,0.08,0.05,0.01,0.005]
    else:
        # Erat=1/0.1067
        Erat=1/0.1
        ticks=[np.log10(5),np.log10(3),np.log10(1),np.log10(0.1),np.log10(0.05),np.log10(0.01)]
        ticknames=[5,3,1,0.1,0.05,0.01]

    
    zplot=(zbins[1:]+zbins[:-1])/2
    zwh=np.where(zplot <= 1/a-1)[0]
    zwh=np.append(zwh,zwh[0]-1)
    Emean=np.log10((Edist[:,:,0].sum(axis=0))/nphotbin.sum(axis=0)/me)
    #pdb.set_trace()

    fig,ax=plt.subplots(1,1)
    ax.set_yscale('log')
    # zmean=1/(amean)-1
    #ax.plot(1/aarr-1,Gty,lw=2.5,color=blue,zorder=0,label='AK 2017 Approx.')
    ax.plot(1/adisc-1,NEWODE,lw=2.5,color=red,zorder=1,label='Exact')
    # sc=ax.scatter(zmean[zwh],Gt[zwh],c=Emean[zwh],marker='o',s=12,zorder=3,cmap=plt.get_cmap('viridis'),edgecolors='none')
    #pdb.set_trace()


    test=find_nearest(adisc, amean[zwh[0]])
    #pdb.set_trace()
    #*NEWODE[test]/Gt[zwh[0]]
    ax.scatter(zmean[zwh],Gt[zwh],s=12,zorder=3)
    print(NEWODE[test]/Gt[zwh[0]])
    #pdb.set_trace()
    # cbar=plt.colorbar(sc)
    # cbar.set_label(r'$E_{{\rm mean}}$/$m_e$',fontsize=18)
    # err=np.sqrt(zdist[:,:,1].sum(axis=0)/nphotbin.sum(axis=0)-zmean*zmean)
    # #err=np.sqrt(err)/(amean*amean)
    # _,__,errorlines=ax.errorbar(zmean[zwh],Gt[zwh],xerr=err[zwh],c=Emean[zwh],zorder=2,fmt='',elinewidth=2)
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(ticknames)
    # cbar.ax.tick_params(labelsize=16)
    # errorlines[0].set_color(cbar.to_rgba(Emean[zwh]))
    #errorlines[1].set_color(cbar.to_rgba(Emean[zwh]))
    #ax.set_aspect(10)
    ax.legend(fontsize=18,loc='lower right')

    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel(r"G(a,a')",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(r'$E_{\rm cut}=$%s $(m_e)$, $z_{\rm inj}=$%s, $N_\gamma=$%s' % (int(Ecut*10)/10., int(1/a-1), nphot), fontsize=18)
    fig.tight_layout()
#    plt.savefig(folsav+nam+'.pdf')

    #ax.plot(1/adisc-1,GODE,lw=2,color=green,zorder=1)
    plt.savefig(folsav+'log/'+nam+'test.pdf')

    plt.close(fig)
#plt.show()



