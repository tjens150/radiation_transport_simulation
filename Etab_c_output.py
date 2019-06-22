from __future__ import division
import numpy as np
import pdb as pdb
from math import pi
import pickle as pickle
#import colormaps as cmaps
from scipy import optimize
from mpi4py import MPI
import time

# plt.register_cmap(name='viridis', cmap=cmaps.viridis)
# cmapV=plt.cm.get_cmap('viridis')

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
    return optimize.newton(f,args[1],fprime=fp,args=args,disp=False,maxiter=100)

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

#CONSTANTS everything in meters, seconds, eV
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


zmax=1300.
zmin=1.
zres=np.flip(np.arange(zmin,zmax+1,1.))
ainj=np.exp(np.arange(np.log(1/(zmax+1.)),np.log(1/51.),0.001))
afinal=1/51.
Na=len(ainj)
Nz=len(zres)

Einj=5*me

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

divy=int(Na/size)
strt=rank*divy
if rank == size-1:
    divy=Na-divy*(size-1)

asep=np.zeros([divy,Nz], dtype=float)
L2sep=np.zeros([divy,Nz], dtype=float)

fol='/Users/Acolyte/NYU/Research/rad_trans/Espace/'

for aa in range(divy):
    a=ainj[strt+aa]
    zinj=int(1/a-1)
    zind=int(zmax)-zinj
    if zind != 0:
        zind-=1
    adisc=np.append(a,1/(np.flip(np.arange(zmin,zinj+1,1.))))
    dlna=np.abs((np.log(adisc[1:])-np.log(adisc[:-1])))
    asep[aa,zind]=Einj
    for ev in (np.arange(len(adisc)-1)+1):
        asep[aa,zind+ev]=imptstep(XODEnewt,pXODEnewt,adisc[ev],asep[aa,zind+ev-1],dlna[ev-1])
    L2sep[aa,zind:]=meanLsq(asep[aa,zind:],adisc)/mpc**2
    
    if rank ==0:
        #if not divy % 10:
        print('%s of %s' % (aa*size,Na))


acom=comm.gather(asep,root=0)
L2com=comm.gather(L2sep,root=0)
if rank == 0:
    atot=acom[0]
    L2tot=L2com[0]
    for i in np.arange(size-1)+1:
        atot=np.concatenate((atot,acom[i]),axis=0)
        L2tot=np.concatenate((L2tot,L2com[i]),axis=0)
    with open(fol+'zspace_E%s.pkl' % int(Einj/me), "wb") as f:
        pickle.dump(atot,f)
        pickle.dump(L2tot,f)
        pickle.dump(ainj,f)
        pickle.dump(zres,f)
