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


def epsH(Ei,a):
    return 1/(sqrtomegaM*H0*a**(3/2.))*nh0*c*((3*me*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*Ei**2))

def epsHtmp(Ei,a):
    return (3*c*me*nh0*sigT*((2*Ei*(-10*Ei**4 + 51*Ei**3*me + 93*Ei**2*me**2 + 51*Ei*me**3 + 9*me**4))/(3.*(2*Ei + me)**3) + 2*(Ei - 3*me)*(Ei + me)*np.arctanh(Ei/(Ei + me))))/(8.*a**1.5*Ei**2*H0*Sqrt(omegaM))

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

def turn(Ei,a):
    return epsH(Ei/a,a)-Ei/a

def turnp(Ei,a):
    return deps(Ei/a,a)-1/a

def newtraph(f,fprime,a,b, ad):
    epsilon=1E-8
    #pdb.set_trace()
    while np.abs(b-a) > epsilon: # I just wanted to add some bisectioning
        bi=(b-a)/2.
        if f(a,ad) > f(b,ad):
            if f(bi+a,ad) >= 0:
                a=bi+a
            else:
                b=b-bi
        else:
            if f(bi+a,ad) >= 0:
                b=b-bi
            else:
                a=bi+a
    #print(a, b, bi)
    # x=max([a,b])
    # val=f(x,ad)
    # epsilon=1E-10
    # pdb.set_trace()
    # while np.abs(val) > epsilon:
    #     x=x-f(x,ad)/fprime(x,ad)
    #     if x >= b or x <=a:
    #        print('Root may not be in given range')
        # val=f(x,ad)
    return (a+b)/2.


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

Na=1000
# Elist=np.array([20,15,10,5,1,0.1,0.01])*me
a=1/1201.
afinal=1/51.
# dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
ad=np.exp(np.linspace(np.log(a),np.log(afinal),Na))

turnover=np.zeros(Na)
Xguess=10.**2
rng=[0.01,2000.] #range from visual inspection of where f is neg and pos for all z
for i in range(Na):
    #turnover[i]=optimize.newton(turn, Xguess, fprime=turnp,args=(ad[i],0))
    turnover[i]=newtraph(turn,turnp,rng[0],rng[1],ad[i])
    if not i% 100:
        print i

fol='/Users/Acolyte/NYU/Research/rad_trans/analytic/'
with open(fol+'XODE1200.pkl', "rb") as f:
    Xdic=pickle.load(f)
    adisc=pickle.load(f)
# with open(fol+'XODE1200_low.pkl', "rb") as f:
#     Xdiclow=pickle.load(f)
#     adisclow=pickle.load(f)
   
fig,ax=plt.subplots(1,1)
for val in Xdic.values():
    ax.plot(1/adisc-1,val*adisc,lw=4,color=purple,zorder=1)
# for val in Xdiclow.values():
#     ax.plot(1/adisclow-1,val*adisclow,lw=4,color=green,zorder=0)
cut=turnover >0.02
ax.plot(1/ad[cut]-1,turnover[cut],lw=4,color=orange,zorder=2,ls='--',label=r'$HE=\dot{\xi}(E)$')
ax.legend(fontsize=18)
ax.invert_xaxis()
ax.set_yscale('log')
ax.set_ylabel('aE (eV)',fontsize=20)
ax.set_xlabel('z',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('XODE.pdf')
plt.show()
