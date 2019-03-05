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

def diffcross(Ei, Ef):
    return (3/8.)*sigT*me/(Ei*Ei)*(Ef/Ei+Ei/Ef-1+(1+me/Ei-me/Ef)**2)

def eps(Ei,a):
    return a**(-3/2.)*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def modeps(Ei,a):
    Earr=nh0*c*(1/Ei)*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return np.einsum('i,j->ij',Earr,1/(H0*np.sqrt(omegaM)*a**(9/2.)))

def flux(u,E,a,epsarr):
    return -np.sqrt(a)/(H0*np.sqrt(omegaM))*(H0*np.sqrt(omegaM)*E/a**(3/2.)+epsarr)*u

def dflux(E,a,epsarr):
    return -np.sqrt(a)/(H0*np.sqrt(omegaM))*(H0*np.sqrt(omegaM)*E/a**(3/2.)+epsarr)

def FUP(fin,fpin,uin):
    aj=np.divide(fin[1:]-fin[:-1],(uin[1:]-uin[:-1]),dtype='float64')
    infind=np.where(~np.isfinite(aj))[0]
    aj[infind]=fpin[infind]
    if ~np.all(da < np.abs(dE/aj)):
        pdb.set_trace()
    return .5*(fin[1:]+fin[:-1])-.5*np.abs(aj)*(uin[1:]-uin[:-1])

def tstep(uin,marr,dlna,dlnE):
    return uin+dlna/dlnE*((1+marr[1:])-(1+marr[:1]))

def deps(Ei,a):
    return 1/(H0*np.sqrt(omegaM)*a**(3/2.))*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def depsu(rit,a):
    return rit*(a**(3/2.)/(H0*np.sqrt(omegaM))*nh0*c*(1/a**3)*((3*me*sigT*((2*(a0*E0/a)*(-40*(a0*E0/a)**3 + 153*(a0*E0/a)**2*me + 186*(a0*E0/a)*me**2 + 51*me**3))/(3.*(2*(a0*E0/a) + me)**3) - (4*(a0*E0/a)*(-10*(a0*E0/a)**4 + 51*(a0*E0/a)**3*me + 93*(a0*E0/a)**2*me**2 + 51*(a0*E0/a)*me**3 + 9*me**4))/(2*(a0*E0/a) + me)**4 + (2*(-10*(a0*E0/a)**4 + 51*(a0*E0/a)**3*me + 93*(a0*E0/a)**2*me**2 + 51*(a0*E0/a)*me**3 + 9*me**4))/(3.*(2*(a0*E0/a) + me)**3) + (2*((a0*E0/a) - 3*me)*((a0*E0/a) + me)*(-((a0*E0/a)/((a0*E0/a) + me)**2) + 1/((a0*E0/a) + me)))/(1 - (a0*E0/a)**2/((a0*E0/a) + me)**2) + 2*((a0*E0/a) - 3*me)*np.arctanh((a0*E0/a)/((a0*E0/a) + me)) + 2*((a0*E0/a) + me)*np.arctanh((a0*E0/a)/((a0*E0/a) + me))))/(8.*(a0*E0/a)**2) - (3*me*sigT*((2*(a0*E0/a)*(-10*(a0*E0/a)**4 + 51*(a0*E0/a)**3*me + 93*(a0*E0/a)**2*me**2 + 51*(a0*E0/a)*me**3 + 9*me**4))/(3.*(2*(a0*E0/a) + me)**3) + 2*((a0*E0/a) - 3*me)*((a0*E0/a) + me)*np.arctanh((a0*E0/a)/((a0*E0/a) + me))))/(4.*(a0*E0/a)**3)))

def rk4(rit,h,t,f):
   ts=len(t)
   r=np.zeros(ts)
   count=0
   for tt in t:
       r[count]=rit
       k1=h*f(rit,tt)
       k2=h*f(rit+0.5*k1,tt+0.5*h)
       k3=h*f(rit+0.5*k2,tt+0.5*h)
       k4=h*f(rit+k3,tt+h)
       rit=rit+(k1+2*k2+2*k3+k4)/6
       count+=1
   return r

def tmp():
    farr=deps(E*ain/np.exp(t),np.exp(t))
    farrhalf=deps(E*ain/np.exp(t+0.5*h),np.exp(t+0.5*h))
    def rk44(rit,h,ts,j):
        r=np.zeros([ts,len(rit)])
        # pdb.set_trace()
        for i in (j+ts):
            r[i,:]=rit
            k1=h*rit*farr[i]
            k2=h*(rit+0.5*k1)*farrhalf[i]
            k3=h*(rit+0.5*k2)*farrhalf[i]
            k4=h*(rit+k3)*farr[i+1]
            rit=rit+(k1+2*k2+2*k3+k4)/6
        return r

def adap(rit,h,t,f,epsilon,E):
    #ts=len(t)
    hcount=0
    r=[]
    h=[h]
    while t[0]+hcount < t[1]:
        print t[0]+hcount-t[1]
        rho=0
        r.append(rit)
        # half=np.array([0,1,2])
        # double=np.array([0,2])
        while rho < 1:
            tmpt=np.array([t[0]+hcount,t[0]+hcount+h[-1],t[0]+hcount+2*h[-1]])
            r1=rk4(rit,h[-1],np.exp(tmpt),f)
            #pdb.set_trace()
            r2=rk4(rit,2*h[-1],np.exp(tmpt[[0,2]]),f)
            rho=h[-1]*epsilon/np.sqrt( ((r1[-1]-r2[-1])**2).sum()/900)
            pdb.set_trace()
            
            if rho >= 1:
                pdb.set_trace()
                if hcount+h[-1]+t[0] > t[1]:
                    h.append(t[1]-hcount)
                else:
                    hcount+=h[-1]
                    rit=r1[1]
                    if rho < 16:
                        h.append(h*rho**(.25))
                    else:
                        h.append(h*2)
                break
            else:
                h.append(h[-1]*rho**(.25))
    return r,h

def verlet(uin,h,a,f,Ei):
   ts=len(a)
   rlen=len(rit)
   r=np.zeros(ts)
   farr=deps(Ei,a)
   vhalf=.5*h*uin*farr[0]
   rit=uin
   r[0]=uin
   for i in np.arange(ts-1)+1:
       x=rit+h*vhalf
       k=h*x*farr[i]
       vhalf=vhalf+k
       rit=x
       r[i]=rit
   return r

# def anaODE(a0,a,E0):
#     def tmpf(a0,a,E0):
#         return (1/(np.sqrt(omegaM)*H0)*me*sigT*(3*a**4*me*(5*a0*E0 + 12*a*me)*np.arctanh((a0*E0)/(a0*E0 + a*me)) + a0*E0*(21*a**3*a0*E0 + (132*a*a0**3*E0**3)/me**2 - (33*a**2*a0**2*E0**2)/me - 36*a**4*me - (80*a0**5*E0**5*(40*a0**2*E0**2 + 48*a*a0*E0*me + 15*a**2*me**2))/(me**3*(2*a0*E0 + a*me)**3) - (504*a0**4*E0**4*np.log(2*a0*E0 + a*me))/me**3)))/(80.*a0**3*E0**3)
#     return tmpf(a0,a,E0)-tmpf(a0,a0,E0)

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps*a**(3/2.)/(sqrtomegaM*H0))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def pXODEnewt(Ei,a, X, h):
    return 1+h-h*deps(Ei,a)

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args)



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
denom=np.einsum('i,j->ij',dR,dt)




fil='comp/ardep_comp_v2.pkl'

with open(fol+fil,"rb") as f:
    ardep=pickle.load(f)
    adist=pickle.load(f)
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
    fig,ax=plt.subplots(1,1)
    E=Estat[nam][0]
    nphot=totnum[nam]
    a=astat[nam][0]
    zwh=np.where(zbins <= 1/a-1)[0]

    NE=1000
    Na=1000
    dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
    minE=np.log(E*10**(-5))
    dlnE=np.abs(np.log(E)-minE)/NE
    pad=2
    #    Edisc=np.linspace(minE-(pad/2*dE),np.log(E)+(pad/2*dE),NE+pad)
    Edisc=np.exp(np.linspace(minE,np.log(E),NE).astype('float64'))
    # maxa=dflux(Edisc[1:-1],a,eps(Edisc[1:-1],a)).max()
    # if da > np.abs(dE/maxa):
    #     da=dE*maxa
    #     Na=np.ceil(np.abs((afinal-a)/da))
        
    adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
    print('Beginning PDE solver...')
    ttot=time.time()
    PDErho=np.zeros(Na)
    PDErho[0]=np.sqrt(a)*nphot/(H0*np.sqrt(omegaM))
    Xstep=np.zeros(Na)
    Xstep[0]=E
    for ev in (np.arange(Na-1)+1):
        Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)
        #PDErho[ev]=PDErho[ev-1]/(1-dlna*deps(Xstep[ev],adisc[ev]))
        if not ev % 10000:
            print(ev)
            
    print('Finished PDE solver.')
    print('PDE took: %s seconds' % (time.time()-ttot))
    #PDErho=PDErho*eps(Xstep,adisc)/(adisc*adisc)
    #PDErho=dlnE*(uin[1:-1,:]*epsarr).sum(axis=0)/(adisc*adisc*adisc)

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
        ticks=[np.log10(0.1),np.log10(0.08),np.log10(0.05),np.log10(0.01),np.log10(0.005)]
        ticknames=[0.1,0.08,0.05,0.01,0.005]
    else:
        Erat=1/0.1067
        ticks=[np.log10(5),np.log10(3),np.log10(1),np.log10(0.1),np.log10(0.05),np.log10(0.01)]
        ticknames=[5,3,1,0.1,0.05,0.01]

    Gty,aarr=yacine(a,astart,Erat)

    Gt=Gt*nphot*E
    Gty=Gty*nphot*E
    zplot=(zbins[1:]+zbins[:-1])/2
    zwh=np.where(zplot <= 1/a-1)[0]
    zwh=np.append(zwh,zwh[0]-1)
    amean=(adist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)
    zmean=amean
    Emean=np.log10((Edist[nam][:,:,0].sum(axis=0))/nphotbin[nam].sum(axis=0)/me)
    # zmean=1/(amean)-1
    ax.plot(1/aarr-1,Gty,lw=2,color=blue,zorder=0)
    #ax.plot(1/adisc-1,PDErho,lw=2,color=green,zorder=1)
    sc=ax.scatter(zmean[zwh],Gt[zwh],c=Emean[zwh],marker='o',s=15,zorder=3,cmap='viridis')
    cbar=plt.colorbar(sc)
    cbar.set_label(r'$E_{{\rm mean}}$/$m_e$')
    err=np.sqrt(adist[nam][:,:,1].sum(axis=0)/nphotbin[nam].sum(axis=0)-amean*amean)
    #err=np.sqrt(err)/(amean*amean)
    _,__,errorlines=ax.errorbar(zmean[zwh],Gt[zwh],xerr=err[zwh],c=Emean[zwh],zorder=2,fmt=None, cerror='black')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticknames)
    errorlines[0].set_color(cbar.to_rgba(Emean[zwh]))
    #errorlines[1].set_color(cbar.to_rgba(Emean[zwh]))
    #ax.set_aspect(10)
    ax.set_xlabel('z')
    ax.set_ylabel(r'G (s$^{-1}$)')
    ax.set_title(r'$E_i=$%s $(m_e)$, $z_i=$%s, $N_\gamma=$%s' % (int(E/me*1000)/1000., int(1/a-1), nphot))
    #ax.set_yscale('log')
    plt.savefig(fol+'/figures/GcompPDE/Gdep_comp_PDE'+nam+'.pdf')
    plt.close(fig)
#plt.show()



