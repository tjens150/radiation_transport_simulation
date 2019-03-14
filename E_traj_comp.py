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
    return a**(-3/2.)*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def deps(Ei,a):
    return 1/(H0*np.sqrt(omegaM)*a**(3/2.))*nh0*c*((3*me*sigT*((2*Ei*(-40*Ei**3 + 153*Ei**2*me + 186*Ei*me**2 + 51*me**3))/(3.*(2*Ei + me)**3) - (4*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(2*Ei + me)**4 + (2*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + (2*(Ei - 3*me)*(Ei + me)*(-(Ei/(Ei + me)**2) + 1/(Ei + me)))/(1 - Ei**2/(Ei + me)**2) + 2*(Ei - 3*me)*np.arctanh(Ei/(Ei + me)) + 2*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2) - (3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(4.*Ei**3))

def XODE(Ei,a):
    Eps=nh0/(a**3)*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))
    return -(Ei+Eps*a**(3/2.)/(sqrtomegaM*H0))

def XODEnewt(Ei,a, X, h):
    return Ei-h*XODE(Ei,a)-X

def pXODEnewt(Ei,a, X, h):
    return 1+h+h*deps(Ei,a)

def imptstep(f,fp, *args):
    return optimize.newton(f,args[1],fprime=fp,args=args)


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


Na=100000
Elist=np.array([5])*me
a=1/1301.
afinal=1/51.
dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
# for Eit in Elist:
#     print(Eit/me)
#     Xstep=np.zeros(Na)
#     Xstep[0]=Eit
#     for ev in (np.arange(Na-1)+1):
#         Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)

            
# fol='/Users/Acolyte/NYU/Research/rad_trans/analytic/'
# with open(fol+'XODE1200.pkl', "rb") as f:
#     Xdic=pickle.load(f)
#     adisc=pickle.load(f)

ode=Xstep
fig,ax=plt.subplots(1,1)
ax.plot(1/adisc-1,ode*adisc,lw=4,color=purple,zorder=1,label=r'$E(t,z_{\rm inj}=1300,E_{\rm inj}=5 m_E)$')
for i in range(200):
    ax.plot(1/mastera[i]-1,masterE[i]*mastera[i],lw=0.5,color=orange,zorder=0)
ax.plot(None,None,lw=3,color=orange,zorder=0, label='Simulation')
ax.set_yscale('log')
ax.legend(fontsize=18)
ax.invert_xaxis()
ax.set_ylabel('aE (eV)',fontsize=20)
ax.set_xlabel('z',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('Etraj.pdf')
plt.show()




