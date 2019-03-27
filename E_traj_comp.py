from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle
from mpi4py import MPI
import time
from scipy import optimize

blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)




def eps(Ei,a):
    return a**(-3.)*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

# def stddeveps(Ei,a):
#     return np.sqrt((Ei**3*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/np.sqrt(6)

def stddeveps(Ei,a):
    return 2*np.sqrt(0.6666666666666666)*np.sqrt((Ei**2*((-3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me)))**2)/(8.*Ei**2*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) + (me*sigT*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/8.))/(me*sigT*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))))

def deps(Ei,a):
    return 1/(H0*np.sqrt(omegaM)*a**(3/2.))*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def sigthet(Ei):
    return 3/8.*sigT*(2*me*((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/Ei**3

def dsigthet(Ei):
    return 3/8.*sigT*((2*me*((Ei*(3*Ei**2 + 18*Ei*me + 8*me**2))/(2*Ei + me)**2 - (4*Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**3 + (Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3)/(2*Ei + me)**2 + ((Ei**2 - 2*Ei*me - 2*me**2)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + (2*Ei - 2*me)*np.arctanh(Ei/(Ei + me))))/Ei**3 - (6*me*((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/Ei**4)

# def stddevdeps(Ei,a):
#     return ((Ei**3*((6*(Ei - 2*me - (4*me**2)/Ei))/((1 + (2*Ei)/me)*me) + (-340*Ei**4 + 736*Ei**3*me + 1698*Ei**2*me**2 + 988*Ei*me**3 + 180*me**4)/(2*Ei + me)**4 - (8*(-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5))/(2*Ei + me)**5 + 3*(1 + (4*me**2)/Ei**2)*np.log(1 + (2*Ei)/me)))/((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))) - (Ei**3*((Ei*(3*Ei**2 + 18*Ei*me + 8*me**2))/(2*Ei + me)**2 - (4*Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**3 + (Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3)/(2*Ei + me)**2 + ((Ei**2 - 2*Ei*me - 2*me**2)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + (2*Ei - 2*me)*np.arctanh(Ei/(Ei + me)))*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me)))**2 + (3*Ei**2*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/(2.*np.sqrt(6)*np.sqrt((Ei**3*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me)))))


def stddevdeps(Ei,a):
    return (np.sqrt(0.6666666666666666)*((Ei**2*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me)))**2*((2*Ei)/((1 + (2*Ei)/me)*me) - (8*Ei**2*(Ei + me))/(2*Ei + me)**3 + (2*Ei**2)/(2*Ei + me)**2 + (4*Ei*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(Ei*(1 - Ei**2/(Ei + me)**2)) - (4*me*np.arctanh(Ei/(Ei + me)))/Ei + (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei**2 + np.log(1 + (2*Ei)/me)))/(8.*Ei**2*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))**2) - (3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me)))*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**2*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) + (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me)))**2)/(4.*Ei**3*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) + (me*sigT*((6*(Ei - 2*me - (4*me**2)/Ei))/((1 + (2*Ei)/me)*me) + (-340*Ei**4 + 736*Ei**3*me + 1698*Ei**2*me**2 + 988*Ei*me**3 + 180*me**4)/(2*Ei + me)**4 - (8*(-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5))/(2*Ei + me)**5 + 3*(1 + (4*me**2)/Ei**2)*np.log(1 + (2*Ei)/me)))/8.))/(me*sigT*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) - (Ei**2*((2*Ei)/((1 + (2*Ei)/me)*me) - (8*Ei**2*(Ei + me))/(2*Ei + me)**3 + (2*Ei**2)/(2*Ei + me)**2 + (4*Ei*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(Ei*(1 - Ei**2/(Ei + me)**2)) - (4*me*np.arctanh(Ei/(Ei + me)))/Ei + (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei**2 + np.log(1 + (2*Ei)/me))*((-3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me)))**2)/(8.*Ei**2*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) + (me*sigT*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/8.))/(me*sigT*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))**2) + (2*Ei*((-3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me)))**2)/(8.*Ei**2*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) + (me*sigT*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/8.))/(me*sigT*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me)))))/np.sqrt((Ei**2*((-3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me)))**2)/(8.*Ei**2*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))) + (me*sigT*((-68*Ei**5 + 184*Ei**4*me + 566*Ei**3*me**2 + 494*Ei**2*me**3 + 180*Ei*me**4 + 24*me**5)/(2*Ei + me)**4 + 3*(Ei - 2*me - (4*me**2)/Ei)*np.log(1 + (2*Ei)/me)))/8.))/(me*sigT*(4*me + (2*Ei**2*(Ei + me))/(2*Ei + me)**2 - (4*me*(Ei + me)*np.arctanh(Ei/(Ei + me)))/Ei + Ei*np.log(1 + (2*Ei)/me))))

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps*a**(3/2.)/(sqrtomegaM*H0))

def XODEsignewt(Ei,a, X, h,pm):
    thisstd=stddeveps(Ei,a)*sigthet(Ei)*nh0*a**(-3.)*c
    return Ei+h*(Ei+(eps(Ei,a)+pm*thisstd)*a**(3/2.)/(sqrtomegaM*H0))-X

def pXODEsignewt(Ei,a, X, h,pm):
    thisstd=(stddevdeps(Ei,a)*sigthet(Ei)+stddeveps(Ei,a)*dsigthet(Ei))*nh0*a**(-3.)*c
    return 1+h*(1+deps(Ei,a)+(pm*thisstd)*a**(3/2.)/(sqrtomegaM*H0))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def pXODEnewt(Ei,a, X, h):
    return 1+h+h*deps(Ei,a)

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args,disp=False)


#CONSTANTS
me=.511*10**6
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
sqrtomegaM=np.sqrt(omegaM)
#CONSTANTS


fol='/Users/Acolyte/NYU/Research/rad_trans/full5000Photons/'
Eread1=[5] #m_e
Eread2=[]
Emax=max(Eread1) #m_e
Emin=0.05 #m_e
zread1=[1300]
#zread1=[1300,1250,1150,1000,900,800,700,600,500,400]
#zread2=[1300,1200,900,800,700,600,500,400]
zread2=[]
thresh=500 #at least this many photons given an astep bin

astart=1/(max(zread1)+1)
afinal=1/51.
astep=0.1
nEbins=3
rstep=0.5 #mpc
rmax=100000 #mpc

alogbins=np.arange(np.log(astart),np.log(afinal),astep)
Ebins=np.linspace(me*Emax,me*Emin,nEbins)
Ebins=[15*me,10*me,5*me, 1.1*me,0.11*me,0.05*me]

nrbins=50
logr=np.linspace(np.log(0.1),np.log(500),nrbins)
rbins=np.insert(np.exp(logr),0,0)
rbins=np.append(rbins,100000)
#nrbins=int(rmax/rstep)

x=[]
y=[]
z=[]
a=[]
E=[]
dE=[]
num=[]
xdic={}
ydic={}
zdic={}
rdic={}
adic={}
Edic={}
dEdic={}
zEdic={}
count=0
print('Loading in data...')
numdata=5000
for zcol in range(2):
    for zi in ([zread1,zread2])[zcol]:
        Etmpread=([Eread1,Eread2])[zcol]
        for Ei in Etmpread:
            fil='%s/fullpar%s_E%s_%s.pkl' % (zi,zi,Ei,numdata)
            with open(fol+fil,"rb") as f:
                masterx=pickle.load(f)
                mastery=pickle.load(f)
                masterz=pickle.load(f)
                mastera=pickle.load(f)
                masterE=pickle.load(f)
        
            tmpdE=[]
            for i in range(len(masterE)):
                ascale=mastera[i][:-1]/mastera[i][1:]
                deltE=masterE[i][:-1]*ascale-masterE[i][1:]
                deltE=np.append(0,deltE)
                tmpdE.append(deltE)
            #reparr=[len(item) for item in mastera]
            # num.append(np.repeat(np.arange(numdata)+count*numdata,reparr))
            # x.append(np.concatenate(masterx))
            # y.append(np.concatenate(mastery))
            # z.append(np.concatenate(masterz))
            # a.append(np.concatenate(mastera))
            # E.append(np.concatenate(masterE))
            # dE.append(np.concatenate(tmpdE))
            count+=1
            xdic['z%s_E%s' % (zi,Ei)]=masterx
            ydic['z%s_E%s' % (zi,Ei)]=mastery
            zdic['z%s_E%s' % (zi,Ei)]=masterz
            rdic['z%s_E%s' % (zi,Ei)]=[np.sqrt(masterx[k]**2+mastery[k]**2+masterz[k]**2) for k in range(len(masterx))]
            adic['z%s_E%s' % (zi,Ei)]=mastera
            Edic['z%s_E%s' % (zi,Ei)]=masterE
            dEdic['z%s_E%s' % (zi,Ei)]=tmpdE
            #numdic['z%s_E%s' % (zi,Ei)]=np.repeat(np.arange(numdata)+count*numdata,reparr)
            zEdic['z%s_E%s' % (zi,Ei)]=[1/(1+zi),Ei*me]


with open('/Users/Acolyte/NYU/Research/rad_trans/xodestd_E5_z1300.pkl') as f:
    Xstep=pickle.load(f)
    Xstepp=pickle.load(f)
    Xstepm=pickle.load(f)

Na=10000
Elist=np.array(Eread1[0])*me
a=1/(np.array(zread1[0])+1.)
afinal=1/51.
dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
# for Eit in Elist:
#     print(Eit/me)
#     Xstep=np.zeros(Na)
#     Xstepp=np.zeros(Na)
#     Xstepm=np.zeros(Na)
#     Xstep[0]=Eit
#     Xstepp[0]=Eit
#     Xstepm[0]=Eit
#     for ev in (np.arange(Na-1)+1):
#         Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)
#         Xstepp[ev]=imptstep(XODEsignewt,pXODEsignewt,adisc[ev],Xstepp[ev-1],dlna,1.)
#         Xstepm[ev]=imptstep(XODEsignewt,pXODEsignewt,adisc[ev],Xstepm[ev-1],dlna,-1.)
#         if not ev%1000:
#             print ev

            
# fol='/Users/Acolyte/NYU/Research/rad_trans/analytic/'
# with open(fol+'XODE1200.pkl', "rb") as f:
#     Xdic=pickle.load(f)
#     adisc=pickle.load(f)

ode=Xstep
fig,ax=plt.subplots(1,1)
ax.plot(1/adisc-1,ode*adisc,lw=4,color=purple,zorder=1,label=r'$E(t,z_{\rm inj}=%s,E_{\rm inj}=%s m_E)$' % (zread1[0],Eread1[0]))
ax.plot(1/adisc-1,Xstepp*adisc,lw=4,color=purple,zorder=1,linestyle='--')
ax.plot(1/adisc-1,Xstepm*adisc,lw=4,color=purple,zorder=1,linestyle='--')
for i in np.arange(500)+2000:
    pltE=np.repeat(masterE[i],2)[:-2]
    plta=np.repeat(mastera[i],2)
    ax.plot(1/plta[1:-1]-1,pltE*plta[:-2],lw=0.5,color=orange,zorder=0)



ax.plot(None,None,lw=3,color=orange,zorder=0, label='Simulation')
ax.set_yscale('log')
ax.legend(fontsize=18)
ax.invert_xaxis()
ax.set_ylabel('aE (eV)',fontsize=20)
ax.set_xlabel('z',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('Etraj_std_E%s_z%s.pdf' % (Eread1[0],zread1[0]))
plt.show()

