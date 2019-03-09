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

Na=100000
Elist=np.array([20,15,10,5,1,0.1,0.01])*me
a=1/1201.
afinal=1/51.
dlna=np.abs((np.log(afinal)-np.log(a))/np.float(Na))
adisc=np.exp(np.linspace(np.log(a),np.log(afinal),Na))
Xdic={}
for Eit in Elist:
    print(Eit/me)
    Xstep=np.zeros(Na)
    Xstep[0]=Eit
    for ev in (np.arange(Na-1)+1):
        Xstep[ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],Xstep[ev-1],dlna)
    Xdic['E%s' % (Eit/me)]=Xstep

fol='/Users/Acolyte/NYU/Research/rad_trans/analytic/'
fil='XODE1200v2.pkl'
with open(fol+fil, "wb") as f:
    pickle.dump(Xdic,f)
    pickle.dump(adisc,f)
