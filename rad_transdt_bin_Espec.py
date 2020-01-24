from __future__ import division
import numpy as np
import pdb as pdb
from math import pi
import pickle as pickle
from mpi4py import MPI
import sys
from _pulla_cy import pulla
from scipy.stats import binned_statistic_2d
from scipy import interpolate
import time

blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)

def maxPT(Ei): #max of the pdf of angle scattered is forward for any Ei.
    return pdfthet(1, Ei)

#acceptance rejection method of sampling from the pdf f, in a domain of [a,b].
#g is max(f) in the domain.
def rej(a,b,f,g,*args): 
    px=0.
    y=1.
    thismax=g(*args)
    while y >=px:  
        x=np.random.uniform(a,b)
        y=np.random.uniform(0,g(*args))
        px=f(x,*args)
    return x

def diffthet(cthet,Ei): #unnormalized cosine of angle scattered PDF as a function of initial energy
    return 3/8.*sigT*(-1 + cthet**2 + 1/(Ei*(1/Ei - (-1 + cthet)/me)) + Ei*(1/Ei - (-1 + cthet)/me))/(Ei**2*(1/Ei - (-1 + cthet)/me)**2)

def sigthet(Ei): #diffthet integrated over angle (or Ef)
    return 3/8.*sigT*(2*me*((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/Ei**3

def pdfthet(cthet,Ei): #normalized pdf to sample the angle scattered
    return diffthet(cthet,Ei)/sigthet(Ei)

#Rotation matrix that takes the basis vector 
#[0,0,1]->[cos(phi)*sin(theta),sin(theta)*sin(phi),cos(theta)]
#i.e., rotates coordinates w.r.t z axis by phi, then rotates w.r.t the y axis by theta.
#thus applying this to the vector does the reverse.
def rotmat(thet,phi): 
    return [[np.cos(phi)*np.cos(thet),-np.sin(phi),np.sin(thet)*np.cos(phi)],
            [np.cos(thet)*np.sin(phi),np.cos(phi),np.sin(phi)*np.sin(thet)],
            [-np.sin(thet),0,np.cos(thet)]]



##
## Assuming matter and radiation dominate
## and that photons interact with ALL electrons regardless if free or bound
## because the photons are comparable to electron rest mass
##

#CONSTANTS all in EV, meters, seconds
me=.511*10**6
zfinal=50.
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
h=0.67
aeq=4.15e-5/(omegaM*h**2)



#COEFFICIENCTS FOR EFFICIENCY
ptmpco=c*nh0/(np.sqrt(omegaM)*H0) #used for the C-code
Lstepco=c*2/(H0*np.sqrt(omegaM))   #used to compute comoving length

    

#Parallel computing variables
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

#Number of photons to run for each a_inj
Ntot=200000*20

#divide for each CPU
N=int(Ntot / size)

#a_inj array, len(alistiter) is the number of simulations with Ntot photons each
astart=1/1501.
aliststep=0.1
alistiter=np.exp(np.arange(np.log(astart),np.log(1/(zfinal+50)),aliststep))

#2d bin initialization
astep=0.1
astartb=1/1501.
alogbins=np.arange(np.log(astartb),np.log(1/(zfinal+1)),astep)
abins=np.exp(alogbins)

nrbins=70
logr=np.linspace(np.log(1),np.log(600),nrbins)
rbins=np.insert(np.exp(logr),0,0)
rbins=np.append(rbins,100000)

# #Load in temperature of black hole (depreciated, just use a flat T=.447*me )
# mbh=10
# T_file='PBH_MBH_%s_L_T.dat' % (mbh)
# ldf=np.loadtxt(T_file)
# Tfarr=ldf[:-1,3]
# zarr=ldf[:-1,0]
# Tf=interpolate.interp1d(np.log(1/(1+zarr)),Tfarr,kind='cubic')

#where I store the pickle file
pikfol='mbh10_rmin1_ainjstep0.1/'

#Set the energy spectrum
Ecut=.447
Eeps=0.045 #2pi*finestructure*me is when bohr radius is comparable to photon wavelength

for astart in alistiter:
    #Initialize the bin arrays to store relevant info,
    #the third dimension stores squares of the values
    result=np.zeros([len(rbins)-1,len(alogbins)-1,2]) #delta E
    photcount=np.zeros([len(rbins)-1,len(alogbins)-1]) #number of scatterings in each bin
    asum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
    zsum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
    rsum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
    Esum=np.zeros([len(rbins)-1,len(alogbins)-1,2]) #Energy of each photon BEFORE scattering
    Etot=0.
    ttot=time.time()
    for i in range(N):
        #sample the energy from a flat energy spectrum
        samp=np.random.uniform()
        Ei=Ecut**samp/Eeps**(samp-1)*me

        #Initialize the injection of the photon
        phistep=np.random.uniform(0,2*pi)
        thetastep=np.random.uniform(0,pi)
        xlist=[0]
        ylist=[0]
        zlist=[0]
        dElist=[0]
        Elist=[Ei]
        alist=[astart]
        Eistep=np.copy(Ei)
        Efstep=np.copy(Ei)
        count=0
        ai=np.copy(astart)
        warning=0
        rot=1.
        
        #While loop that propagates the photon and records data
        while 1/ai-1 > zfinal:
            astep=np.copy(ai)
            seed=np.random.randint(0,2147483647) #seed for the C code

            #Cythonized Python code (C), evolves through time checking if the photon has scattered
            #outputs the time it scattered, and what the energy was after the redshifting.
            astep,Eistep=pulla(astep,Eistep,zfinal,ptmpco,aeq,me,sigT,seed) 

            if 1/astep-1 < zfinal:
                warning = 1
                break

            #Comoving length with rad+mat
            Lstep=Lstepco*((aeq+astep)**(1/2.)-(aeq+ai)**(1/2.))
            
            #the 3d vector of the photon
            sth=np.sin(thetastep)
            cth=np.cos(thetastep)
            vec=[[sth*np.cos(phistep)*Lstep],
                 [sth*np.sin(phistep)*Lstep],
                 [cth*Lstep]]

            #The vector is v.Rz_1.Ry_1.Rz_2.Ry_2...->v.rot after N scatters,
            #thus I need to apply the reverse to get v,
            #the desired vector in my original coordinate system
            dotvec=np.dot(rot,vec).flatten()

            #Sample the new angle in which the photon scatter (equivalently the new energy)
            phistep=np.random.uniform(0,2*pi)
            thetastep=np.arccos(rej(-1,1,pdfthet,maxPT,Eistep))
            Efstep=1/(2/me*np.sin(thetastep/2.)**2+1/Eistep)

            #Combine the previous rotation matrix with the new one.
            rot=np.dot(rot,rotmat(thetastep,phistep))

            #store info and set up for next iteration
            xlist.append((dotvec[0]+xlist[-1]))
            ylist.append((dotvec[1]+ylist[-1]))
            zlist.append((dotvec[2]+zlist[-1]))
            dElist.append(Eistep-Efstep)
            Elist.append(Efstep)
            alist.append(astep)
            Eistep=Efstep
            ai=astep
            count+=1

        #if photons do not scatter at all by z=50, then break
        if len(xlist) < 2:
            break

        #a photon has been propagated, bin all its information
        Etot+=Ei

        #the first element is the superfluous initialization
        newr=(np.sqrt(np.array(xlist)**2+np.array(ylist)**2+np.array(zlist)**2)/mpc)[1:]
        aph=np.array(alist)[1:]
        dEph=np.array(dElist)[1:]
        zaph=1/aph-1

        #Energy BEFORE the photon scatters
        newE=np.array(Elist)[:-1]

        #Sum of the quantities
        photcount+=np.array(binned_statistic_2d(newr,aph,np.ones_like(newr,dtype='int'),statistic='sum', bins=[rbins,abins])[0])
        result[:,:,0]+=np.array(binned_statistic_2d(newr,aph,dEph,statistic='sum', bins=[rbins,abins])[0])
        asum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,aph,statistic='sum', bins=[rbins,abins])[0])
        zsum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,zaph,statistic='sum', bins=[rbins,abins])[0])
        rsum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,newr,statistic='sum', bins=[rbins,abins])[0])
        Esum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,newE,statistic='sum', bins=[rbins,abins])[0])
        
        #Sum of the square of the quantities
        result[:,:,1]+=np.array(binned_statistic_2d(newr,aph,dEph*dEph,statistic='sum', bins=[rbins,abins])[0])
        asum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,aph*aph,statistic='sum', bins=[rbins,abins])[0])
        zsum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,zaph*zaph,statistic='sum', bins=[rbins,abins])[0])
        rsum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,newr*newr,statistic='sum', bins=[rbins,abins])[0])
        Esum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,newE*newE,statistic='sum', bins=[rbins,abins])[0])
        if (rank == 0) & (not i % 1000):
            print(i,'of ',N)
            print(time.time()-ttot)



    #Gather the arrays from all the CPUs
    Etot=comm.gather(Etot,root=0)    
    photcount=comm.gather(photcount,root=0)
    result=comm.gather(result,root=0)
    asum=comm.gather(asum,root=0)
    zsum=comm.gather(zsum,root=0)
    rsum=comm.gather(rsum,root=0)
    Esum=comm.gather(Esum,root=0)
        
    if not rank:
        #Add each CPU's binned statistics together
        mphotcount=np.zeros(np.shape(photcount[0]))
        mresult=np.zeros(np.shape(result[0]))
        masum=np.zeros(np.shape(result[0]))
        mzsum=np.zeros(np.shape(result[0]))
        mrsum=np.zeros(np.shape(result[0]))
        mEsum=np.zeros(np.shape(result[0]))
        mEtot=0.
        for ss in range(len(photcount)):
            mphotcount+=photcount[ss]
            mresult+=result[ss]
            masum+=asum[ss]
            mzsum+=zsum[ss]
            mrsum+=rsum[ss]
            mEsum+=Esum[ss]
            mEtot+=Etot[ss]

        #save the binned statistics
        tit='z'+str(int(1/astart-1))+'_'+'MBH'+str(mbh)+'_N'+str(Ntot)+'_binned_Etot'
        print(tit)
        fil=tit+'.pkl'
        with open('./pickle/'+pikfol+fil, "wb") as f:
            pickle.dump(mphotcount,f)
            pickle.dump(mresult,f)
            pickle.dump(masum,f)
            pickle.dump(mzsum,f)
            pickle.dump(mrsum,f)
            pickle.dump(mEsum,f)
            pickle.dump(mEtot,f)
