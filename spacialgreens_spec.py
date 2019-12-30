from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle
import colormaps as cmaps
from scipy import optimize
from scipy import interpolate

plt.register_cmap(name='viridis', cmap=cmaps.viridis)
cmapV=plt.cm.get_cmap('viridis')

blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)


def HubbleRate(a):
    return H0*np.sqrt(omegaM/a**3+omegaK/a**2+omegaDE+omegaR/a**4)

def trap_rule(N, a, b, f):
    h = (b-a)/N
    arr=h*np.arange(1,N)
    return h*(f(a)*.5+f(b)*.5+f(arr+a).sum())

def cross(Ei):
    return (3/8.)*sigT*me/(Ei*Ei)*(2*(Ei**3 + 9*Ei*Ei*me+8*Ei*me*me+2*me**3)/((2*Ei+me)*(2*Ei+me))+(Ei-2*me-2*me*me/Ei)*np.log(1+2*Ei/me))

def eps(Ei,a):
    return a**(-3.)*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def deps(Ei,a):
    return 1/(HubbleRate(a)*a**(3.))*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def find_nearest(array, value):
    array = np.asarray(array)
    array[~np.isfinite(array)]=0.
    idx = (np.abs(array - value)).argmin()
    return idx

def spacialint(g,dR,dt):
    diff=np.repeat(dR[:,np.newaxis],dt.shape[0],axis=1)
    return (g*diff).sum(axis=0)

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args)#,disp=False)

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps/HubbleRate(a))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def pXODEnewt(Ei,a, X, h):
    return 1+h+h*deps(Ei,a)

def meanLsq(xode,aarr):
    iso=np.zeros(xode.size)
    crossiter=cross(xode)
    ai=aarr[0]
    def tmpf(aint):
        return c/(HubbleRate(aint)*nh0)
    fint=tmpf(aarr)/crossiter
    for i in range(iso.size):
        finttmp=fint[0:i+1]
        atmp=aarr[0:i+1]
        iso[i]=((finttmp[:-1]+finttmp[1:])*(atmp[1:]-atmp[:-1])).sum()/2
    # pdb.set_trace()
    return iso

def newG(adisc,XODEval,E0):
    return (adisc[0]/adisc)**3*eps(XODEval,adisc)/E0

def Edot(Ei,a):
    return nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def Edepf(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    inter=-Eps/a*(a**(3/2.)/(sqrtomegaM*H0))
    retval=np.zeros(len(a))
    for i in range(len(a)):
        tmpinter=inter[0:i+1]
        tmpa=a[0:i+1]
        retval[i]=((tmpinter[:-1]+tmpinter[1:])*(tmpa[1:]-tmpa[:-1])).sum()/2
    return retval

def dtfunc(a1,a2):
    return 2/(3*H0*sqrtomegaM)*(a2*np.sqrt(a2+aeq)-a1*np.sqrt(a1+aeq)+2*aeq*(np.sqrt(a1+aeq)-np.sqrt(a2+aeq)))


#CONSTANTS everything in meters, seconds, eV
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
#CONSTANTS



# nrbins=50
# zread1=[1300,1250,1150,1000,900,800,700,600,500,400]
# logr=np.linspace(np.log(0.1),np.log(500),nrbins)
# rbins=np.insert(np.exp(logr),0,0)
# rbins=np.append(rbins,100000)
# astart=1/(max(zread1)+1)
# afinal=1/51.
# astep=0.1
# alogbins=np.arange(np.log(astart),np.log(afinal),astep)

astep=0.1
astart=1/1301.
afinal=1/51.
alogbins=np.arange(np.log(astart),np.log(afinal),astep)
abins=np.exp(alogbins)

nrbins=70
logr=np.linspace(np.log(20),np.log(600),nrbins)
rbins=np.insert(np.exp(logr),0,0)
rbins=np.append(rbins,100000)

#a32=np.exp(3/2.*alogbins)
#dt=2/3.*(a32[1:]-a32[:-1])/(H0*np.sqrt(omegaM))
amean=(abins[1:]+abins[:-1])/2.
dt=dtfunc(abins[:-1],abins[1:])
dR=4*pi/3*(rbins[1:]**3-rbins[:-1]**3)
# amean=1.

astartb=1/1301.
aliststep=0.05
alistiter=np.exp(np.arange(np.log(astartb),np.log(1/(zfinal+50)),aliststep))


znam=np.asarray(1/alistiter-1, dtype=int)
mbh=100
Nnam=100000

T_file='PBH_MBH_%s_T_novbc.dat' % (mbh)
ldf=np.loadtxt(T_file)
Tfarr=ldf[:-1,-1]
zarr=ldf[:-1,0]
Tf=interpolate.interp1d(np.log(1/(1+zarr)),Tfarr,kind='cubic')


nphotbin={}
ardep={}
adist={}
zdist={}
rdist={}
Edist={}
totnum={}
astat={}
Estat={}

Emin=0.01
nEbins=20.

znam=znam[znam > 450] #redshifts below this caused problems with photons never scattering

for zz in znam:
    fil='pickle/z%s_MBH%s_N%s_binned.pkl' % (zz,mbh,Nnam)

    with open('./'+fil,"rb") as f:
        tnphotbin=pickle.load(f)
        tardep=pickle.load(f)
        tadist=pickle.load(f)    
        tzdist=pickle.load(f)
        trdist=pickle.load(f)
        tEdist=pickle.load(f)
        #    tNscatter=pickle.load(f)

    ttotnum=Nnam*nEbins
    tastat=1/(1.+zz)
    tEstat=Tf(np.log(tastat))*me

    nphotbin['z%s' % (zz)]=tnphotbin
    ardep['z%s' % (zz)]=tardep
    adist['z%s' % (zz)]=tadist
    zdist['z%s' % (zz)]=tzdist
    rdist['z%s' % (zz)]=trdist
    Edist['z%s' % (zz)]=tEdist
    totnum['z%s' % (zz)]=ttotnum
    astat['z%s' % (zz)]=tastat
    Estat['z%s' % (zz)]=tEstat
    #Nscatter={'z%s_E%s' % (zz):tNscatter}

zbins=1/np.exp(alogbins)-1
#Erat=1/0.1



for nam in ardep:
    print(nam)
    Na=10000
    # if nam != 'z1200_E5':
    #     continue
    Ecut=Estat[nam]
    Ebins=np.linspace(Emin,Ecut,nEbins)
    dE=(Ecut-Emin)/nEbins
    Etot=(Ebins*Ntot).sum()
    nphot=totnum[nam]
    a=astat[nam]
    dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
    adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
    
    print('Starting ODE solver...')
    Xstep=np.zeros(Na)
    Xstep[0]=Ecut
    for ev in (np.arange(Na-1)+1):
        Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)
    print('Starting Lsq solver...')
    Lsq=meanLsq(Xstep,adisc)
    Liso=np.sqrt(Lsq)/mpc
    # Edep=-Edepf(Xstep,adisc)
    # Edt=(Edep[1:]-Edep[:-1])/(adisc[1:]-adisc[:-1])
    # Edt=Edot(Xstep,adisc)
    # meanL=(Liso[1:]+Liso[:-1])/2
    # Edtdx=(Edt[1:]-Edt[:-1])/(Liso[1:]-Liso[:-1])/(meanL**2*4*pi)
    # isoint=Edtdx*meanL**2
    # isonorm=((isoint[:-1]+isoint[1:])*(meanL[1:]-meanL[:-1])).sum()*4*pi/2
    # # Edep=newG(adisc,Xstep,E)[1:]*(adisc[1:]-adisc[:-1])/(Liso[1:]-Liso[:-1])
    # # Liso=Liso[1:]
    # # # Edep=E-Xstep
    # Ldom=meanL**3*4*pi
    # isospac=Edtdx/isonorm

    # pdb.set_trace()
    zwh=np.where(zbins <= 1/a-1)[0]
    thisnphot=nphotbin[nam].sum(axis=0)
    zmean=(zdist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    zstd=np.sqrt((zdist[nam][:,:,1].sum(axis=0))/nphotbin[nam].sum(axis=0)-zmean**2)
    amean=(adist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    # Emean=(Edist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    # Evar=np.sqrt((Edist[nam][:,:,1].sum(axis=0))/nphotbin[nam].sum(axis=0)-Emean**2)
    Emean=(Edist[nam][:,:,0])/nphotbin[nam]
    Evar=np.sqrt((Edist[nam][:,:,1])/nphotbin[nam]-Emean**2)
    dEmean=(ardep[nam][:,:,0])/nphotbin[nam]
    dEvar=np.sqrt((ardep[nam][:,:,1])/nphotbin[nam]-dEmean**2)
    rmean=rdist[nam][:,:,0]/nphotbin[nam]
    denom=np.einsum('i,j->ij',dR,dt*(amean/a)**3)
    if zwh[0]-1 > -1:
        # zwhmod=np.insert(zwh,0,zwh[0]-1)
        # tmpdenom=np.insert(denom,zwh[0],dR*(amean[zwh[0]-1]/a)**3*2/3.*(a32[zwh[0]]-a**(3/2.))/(H0*np.sqrt(omegaM)),axis=1)
        tmpamean=(abins[zwh[0]]+a)/2.
        tmpdenom=np.insert(denom,zwh[0],dR*(amean[zwh[0]-1]/a)**3*dtfunc(a,abins[zwh[0]]),axis=1)

        tmpdenom=np.delete(tmpdenom,zwh[0]-1,axis=1)
    else:
        tmpdenom=denom
        #for nam in ['z1266_E0.11']:
    rhodep=ardep[nam][:,:,0]/tmpdenom
    G=rhodep*Ecut/(Etot*HubbleRate(a))
    Gt=spacialint(G,dR,dt)
    zplot=(zbins[1:]+zbins[:-1])/2
    zwh=np.where(zplot <= 1/a-1)[0]
    zwh=np.append(zwh,zwh[0]-1)
    stepres=150
    zplts=-np.arange(-(1/a-1)+100,-(1/afinal-1),stepres)
#    zplts=np.insert(zplts,1,1111)
    rgord=np.linspace(0,1,len(zplts))
    rgba=cmapV(rgord)
    indxarr=np.zeros(len(zplts),dtype='int')
    err=np.sqrt(ardep[nam][:,:,1]/nphotbin[nam]-(ardep[nam][:,:,0]/nphotbin[nam])**2)/tmpdenom/(nphot*E)
    # pdb.set_trace()
    fig,ax=plt.subplots(1,1)
#    ax.plot([],[],lw=0,label=r'($z_{\rm mean}$, $E_{\rm mean}$)')
    # tmpcut=(meanL < 60)
    #ax.plot(meanL[tmpcut],(isospac*Ldom)[tmpcut],lw=2,color=orange,zorder=(len(zplts)+1),linestyle='--')#,label='Isotropic')
    for cc in range(len(zplts)):
        indx=find_nearest(zmean,zplts[cc])
        indxv=find_nearest(zmean[indx],1/adisc-1)
        # indxvp=find_nearest(zmean[indx]+zstd[indx],1/adisc-1)
        # indxvm=find_nearest(zmean[indx]-zstd[indx],1/adisc-1)
        #print('Traveled %s Mpc by z~ %s' % (Liso[indxv],zmean[indx]))
        cut=(nphotbin[nam][:,indx]>100)
        # if cc == 0:
        #     cut=(np.insert(cut,0,False))[:-1]
        # elif cc == len(zplts)-1:
        #     cut=(np.append(cut,False))[1:]
        # ax.axvline(x=Liso[indxvp],lw=1.5,linestyle='-.',zorder=0,color=rgba[cc],alpha=0.5)
        # ax.axvline(x=Liso[indxvm],lw=1.5,linestyle='-.',zorder=0,color=rgba[cc],alpha=0.5)
        
        ##ax.axvline(x=Liso[indxv],lw=2,linestyle=':',zorder=0,color=rgba[cc],alpha=0.85)
        
        # Etest=int(Emean[indx]/me*100)/100.
        # if Etest == 0:
        #     Estr='%.0e' % (Emean[indx]/me)
        # else:
        #     Estr=str(Etest)
        ax.plot(rmean[cut,indx],4*pi*rmean[cut,indx]**3*G[cut,indx]/Gt[indx],lw=2,color=rgba[cc], zorder=cc,label=r'(%s$\leq z\leq$ %s)' % (int(zbins[indx]),int(zbins[indx+1])))
        # ax.plot([],[],lw=0,label=r'($N=%s$)' % (int(thisnphot[indx])))
        unnormgaus=4*pi*Liso**3*np.exp(-3*(Liso)**2/(2*Liso[indxv]**2))
        norm=3/2.*np.sqrt(3/2.)/(pi**(3/2.)*Liso[indxv]**3)
        ax.plot(Liso,unnormgaus*norm,lw=2,color=rgba[cc], linestyle='-.',alpha=0.5, zorder=cc)
        print(r'(%s$\leq z\leq$ %s)' % (int(zbins[indx]),int(zbins[indx+1])))
        # for pp in range(len(Emean[:,0])):
        #     print('(%.2e< r< %.2e), E_mean: %.2e, E_var: %.2e, dE_mean: %.2e, dE_var: %.2e, N_scatter: %s, N_1st: %s, N_2nd: %s' % (rbins[pp],rbins[pp+1],Emean[pp,indx]/me,Evar[pp,indx]/me,dEmean[pp,indx]/me,dEvar[pp,indx]/me,int((nphotbin[nam])[pp,indx]), (Nscatter[nam])[pp,indx,0], (Nscatter[nam])[pp,indx,1]))
        #tmperr=err[:,indx]
        #tmperr[(G[:,indx]-tmperr)<0]=0.
        #_,__,errorlines=ax.errorbar(rmean[:,indx],4*pi*rmean[:,indx]**3*G[:,indx]/Gt[indx],yerr=err[:,indx]/Gt[indx],ecolor=rgba[cc],fmt='none', cerror='black',elinewidth=2,zorder=cc)
        # indxarr[c]=indx
    # cbar=fig.colorbar(axt)
    # cbar.set_ticks(rgord)
    # cbar.set_ticklabels(int(zbins[indx]))
    # cbar.ax.tick_params(labelsize=16)
    ll=ax.legend(fontsize=10,loc='upper left',ncol=1,labelspacing=0.2,handlelength=0.75)
    ll.set_zorder(10000)
    ax.set_xlabel(r'$r$ (Mpc)',fontsize=20)
    ax.set_ylabel(r'$4\pi r^3$G',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(r'$E_{\rm cut}=$%s $(m_e)$, $z_{\rm inj}=$%s, $N_{\rm tot}=$%s' % (int(Ecut/me*1000)/1000., int(1/a-1), nphot), fontsize=18)
    ax.set_xscale('log')
    fig.tight_layout()
    plt.savefig('./figures/spatial_Espec/spec_'+nam+'.pdf')
    # ax.set_yscale('log')
    # fig.tight_layout()
    # ax.legend(fontsize=14,loc='lower right',ncol=3,labelspacing=0.2).set_zorder(10000)
    # # ax.legend(labelspacing=40)
    # plt.savefig(fol+'/figures/log'+nam+'clean.pdf')
    plt.close(fig)

