from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb as pdb
from math import pi
import pickle as pickle
from mpi4py import MPI
import sys
from _pulla_cy import pulla
from scipy.stats import binned_statistic_2d
import time

blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)

def sigp1(En,ai,ap): #this is an ugly mathematica integral of cross(Ei)/a^(5/2) from ai to ap
    return (2*me**3*((-8*np.sqrt(ai)*En**2*(3*En + ai*me))/(me**2*(2*En + ai*me)) + (8*np.sqrt(ap)*En**2*(3*En + ap*me))/(me**2*(2*En + ap*me)) + 12*np.sqrt(2)*(En/me)**2.5*np.arctan(1/(np.sqrt(2)*np.sqrt(En/(ai*me)))) - 12*np.sqrt(2)*(En/me)**2.5*np.arctan(1/(np.sqrt(2)*np.sqrt(En/(ap*me))))) + En**3*((4*En + 3*ai*me)/(np.sqrt(ai)*(2*En + ai*me)) + (-4*En - 3*ap*me)/(np.sqrt(ap)*(2*En + ap*me)) - (3*np.sqrt(me/En)*np.arctan(np.sqrt(2)*np.sqrt(En/(ai*me))))/np.sqrt(2) + (3*np.sqrt(me/En)*np.arctan(np.sqrt(2)*np.sqrt(En/(ap*me))))/np.sqrt(2)) + (9*En**2.5*np.sqrt(me)*(-4*np.sqrt(ai*En**3*me) + 4*np.sqrt(ap*En**3*me) - 2*ap*np.sqrt(ai*En*me**3) + 2*ai*np.sqrt(ap*En*me**3) + np.sqrt(2)*(2*En + ai*me)*(2*En + ap*me)*np.arctan(np.sqrt(2)*np.sqrt(En/(ai*me))) - np.sqrt(2)*(2*En + ai*me)*(2*En + ap*me)*np.arctan(np.sqrt(2)*np.sqrt(En/(ap*me)))))/((2*En + ai*me)*(2*En + ap*me)) + 16*En**1.5*np.sqrt(me)*((2*np.sqrt(ai*En**3*me))/(2*En + ai*me) - (2*np.sqrt(ap*En**3*me))/(2*En + ap*me) + np.sqrt(2)*En*(np.arctan((np.sqrt(2)*En)/np.sqrt(ai*En*me)) - np.arctan((np.sqrt(2)*En)/np.sqrt(ap*En*me)))))/(2.*En**4)

def sigp2(En,ai,ap): #this is an ugly mathematica integral of cross(Ei)/a^(5/2) from ai to ap
    return (2*((-6*En**2)/np.sqrt(ai) + (6*En**2)/np.sqrt(ap) + 8*np.sqrt(ai)*En*me - 8*np.sqrt(ap)*En*me + 3*np.sqrt(2)*np.sqrt(En**3*me)*np.arctan(np.sqrt(2)*np.sqrt(En/(ai*me))) - 3*np.sqrt(2)*np.sqrt(En**3*me)*np.arctan(np.sqrt(2)*np.sqrt(En/(ap*me))) + 4*np.sqrt(2)*np.sqrt(En**3*me)*np.arctan(np.sqrt((ai*me)/En)/np.sqrt(2)) - 4*np.sqrt(2)*np.sqrt(En**3*me)*np.arctan(np.sqrt((ap*me)/En)/np.sqrt(2)) - (3*En**2*np.log(ai))/np.sqrt(ai) - 6*np.sqrt(ai)*En*me*np.log(ai) - 2*ai**1.5*me**2*np.log(ai) + (3*En**2*np.log(ap))/np.sqrt(ap) + 6*np.sqrt(ap)*En*me*np.log(ap) + 2*ap**1.5*me**2*np.log(ap) - (3*En**2*np.log(me))/np.sqrt(ai) + (3*En**2*np.log(me))/np.sqrt(ap) - 6*np.sqrt(ai)*En*me*np.log(me) + 6*np.sqrt(ap)*En*me*np.log(me) - 2*ai**1.5*me**2*np.log(me) + 2*ap**1.5*me**2*np.log(me) + (3*En**2*np.log(2*En + ai*me))/np.sqrt(ai) + 6*np.sqrt(ai)*En*me*np.log(2*En + ai*me) + 2*ai**1.5*me**2*np.log(2*En + ai*me) - (3*En**2*np.log(2*En + ap*me))/np.sqrt(ap) - 6*np.sqrt(ap)*En*me*np.log(2*En + ap*me) - 2*ap**1.5*me**2*np.log(2*En + ap*me)))/(3.*En**3)

def diffthet(cthet,Ei):
    return 3/8.*sigT*(-1 + cthet**2 + 1/(Ei*(1/Ei - (-1 + cthet)/me)) + Ei*(1/Ei - (-1 + cthet)/me))/(Ei**2*(1/Ei - (-1 + cthet)/me)**2)

def sigthet(Ei):
    return 3/8.*sigT*(2*me*((Ei*(Ei**3 + 9*Ei**2*me + 8*Ei*me**2 + 2*me**3))/(2*Ei + me)**2 + (Ei**2 - 2*Ei*me - 2*me**2)*np.arctanh(Ei/(Ei + me))))/Ei**3

def trap_rule(N, a, b, f, *args):
    h = (b-a)/float(N)
    arr=h*np.arange(1,N)
    return h*(f(a,*args)*.5+f(b,*args)*.5+f(arr+a,*args).sum())

def diffcross(Ei, Ef):
    return (3/8.)*sigT*me/(Ei*Ei)*(Ef/Ei+Ei/Ef-1+(1+me/Ei-me/Ef)**2)

def cross(Ei):
    return (3/8.)*sigT*me/(Ei*Ei)*(2*(Ei**3 + 9*Ei*Ei*me+8*Ei*me*me+2*me**3)/((2*Ei+me)*(2*Ei+me))+(Ei-2*me-2*me*me/Ei)*np.log(1+2*Ei/me))

def pdf(Ef, Ei):
    return diffcross(Ei, Ef)/cross(Ei)

def maxPE(Ei):
    maxx=np.array([Ei/(1+2*Ei/me),Ei])
    return np.max(pdf(Ei,maxx))

def maxPL(Ei,ai):
    return pdfL(ai, Ei, ai)

def maxPT(Ei):
    return pdfthet(1, Ei)

def rej(a,b,f,g,*args):
    px=0.
    y=1.
    while y >=px:  
        x=np.random.uniform(a,b)
        y=np.random.uniform(0,g(*args))
        px=f(x,*args)
    return x

def mcmc(px,a,b,f,g,*args):
    xret=np.copy(args[-1])
    pret=np.copy(px)
    x=np.random.uniform(a,b)
    pxnew=f(x,*args)
    y=np.random.uniform(0,g(*args))
    if pxnew >= px or y <= pxnew/px:
        pret=np.copy(pxnew)
        xret=np.copy(x)
    return xret,pret

def pdfL(ap,Ei,ai):
    En=Ei*ai
    quad=sigp1(En,ai,ap)+sigp2(En,ai,ap)
    expon=-1/(H0*np.sqrt(omegaM))*nh0*c*3/8.*sigT*quad*me
    nhf=nh0/ap**3
    return np.exp(expon)*1/(H0*np.sqrt(omegaM))*nhf*c*cross(Ei*ai/ap)*ap**(1/2)

def pdfthet(cthet,Ei):
    return diffthet(cthet,Ei)/sigthet(Ei)

def thomsPDF(ap,ai):
    return np.exp(1/(3/2.*H0*np.sqrt(omegaM))*nh0*c*sigT*(ap**(-3/2.)-ai**(-3/2.)))*(1/(H0*np.sqrt(omegaM))*nh0*c*sigT*ap**(-5/2.))

def thomsCDF(ai):
    x=np.random.uniform(0,1)
    return (np.log(1-x)/(1/(3/2.*H0*np.sqrt(omegaM))*nh0*sigT*c)+ai**(-3/2.))**(-2/3)

def getLOLD(Ei, nh):
    lamb=1/(nh*cross(Ei))
    r=np.random.uniform(0,1)
    return -lamb*np.log(r)

def simp_rule(N, a, b, f, *args):
    if N % 2:                   
        N+1
    h = (b-a)/N
    arreven=h*np.arange(2,N-1,2)
    arrodd=h*np.arange(1,N,2)
    return (1/3.)*h*(f(a, *args)+f(b,*args)+4*f(arrodd+a,*args).sum()+2*f(arreven+a,*args).sum())

# def rotmat(thet,phi):
#     return [[np.cos(phi)*np.cos(thet),-np.sin(phi)*np.cos(thet),np.sin(thet)],
#             [np.sin(phi),np.cos(phi),0],
#             [np.sin(thet)*np.cos(phi),np.sin(phi)*np.sin(thet),np.cos(thet)]]
def rotmat(thet,phi): #corrected maybe FINALLY???
    return [[np.cos(phi)*np.cos(thet),-np.sin(phi),np.sin(thet)*np.cos(phi)],
            [np.cos(thet)*np.sin(phi),np.cos(phi),np.sin(phi)*np.sin(thet)],
            [-np.sin(thet),0,np.cos(thet)]]



#if __name__ =="__main__":

##
## At the moment this is only assuming matter domination
## and that photons interact with ALL electrons regardless if free or bound
## because the photons are comparable to electron rest mass
##

#CONSTANTS
me=.511*10**6
alist=[1/1201.]

zfinal=50.
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
h=0.67
aeq=4.15e-5/(omegaM*h**2)
#CONSTANTS

#COEFFICIENCTS FOR EFFICIENCY
ptmpco=c*nh0/(np.sqrt(omegaM)*H0)
Lstepco=c*2/(H0*np.sqrt(omegaM))    
#COEFFICIENCTS FOR EFFICIENCY
    
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

Eim=[10,5,1]
# divy=int(size/len(Eilist))
# if size == 1:
#     divy=1
# surplus=size-(divy*len(Eilist))
# if size != 1:
#     assert not surplus
Ntot=800000
N=int(Ntot / size)
# Eim=[Eilist[rank % len(Eilist)]]
# color= rank % len(Eilist)
# newcomm=comm.Split(color,rank)
# newrank=newcomm.Get_rank()
# comm.Barrier()

astep=0.1
alogbins=np.arange(np.log(astart),np.log(afinal),astep)
#Ebins=np.linspace(me*Emax,me*Emin,nEbins)
#Ebins=[15*me,10*me,5*me, 1.1*me,0.11*me,0.05*me]
abins=np.exp(alogbins)

nrbins=70
logr=np.linspace(np.log(20),np.log(600),nrbins)
rbins=np.insert(np.exp(logr),0,0)
rbins=np.append(rbins,100000)


for astart in alist:
    for ee in range(len(Eim)):
        Ei=Eim[ee]*me
        afarr=np.zeros(N)
        Narr=np.zeros(N, dtype=int)
        result=np.zeros([len(rbins)-1,len(alogbins)-1,2])
        photcount=np.zeros([len(rbins)-1,len(alogbins)-1])
        asum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
        zsum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
        rsum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
        Esum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
        for i in range(N):
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
            ttot=time.time()
            while 1/ai-1 > zfinal:
                astep=np.copy(ai)
                seed=np.random.randint(0,2147483647)
                astep,Eistep=pulla(astep,Eistep,zfinal,ptmpco,aeq,me,sigT,seed)
                if 1/astep-1 < zfinal:
                    warning = 1
                    break
                Lstep=Lstepco*((aeq+astep)**(1/2.)-(aeq+ai)**(1/2.))
                sth=np.sin(thetastep)
                cth=np.cos(thetastep)
                vec=[[sth*np.cos(phistep)*Lstep],
                     [sth*np.sin(phistep)*Lstep],
                     [cth*Lstep]]
                #this vector is v.Rz_1.Ry_1.Rz_2.Ry_2...->v.rot Thus I need to apply the reverse to get v, the desired vector in my original coordinate system
                dotvec=np.dot(rot,vec)
                xlist.append(dotvec[0]+xlist[-1])
                ylist.append(dotvec[1]+ylist[-1])
                zlist.append(dotvec[2]+zlist[-1])

                phistep=np.random.uniform(0,2*pi)
                thetastep=np.arccos(rej(-1,1,pdfthet,maxPT,Eistep))
                Efstep=1/(2/me*np.sin(thetastep/2.)**2+1/Eistep)
                rot=np.dot(rot,rotmat(thetastep,phistep))
                dElist.append(Eistep-Efstep)
                Eistep=Efstep#*ai/astep
                ai=astep
                Elist.append(Eistep)
                alist.append(astep)
                count+=1
            # afarr[i]=1/ai-1
            # Narr[i]=count
            newr=np.sqrt(np.array(xlist)**2+np.array(ylist)**2+np.array(zlist)**2)/mpc
            aph=np.array(alist)
            dEph=np.array(dElist)
            zaph=1/aph-1
            newE=np.array(Elist)[:1]

            photcount+=np.array(binned_statistic_2d(newr,aph,np.ones_like(newr,dtype='int'),statistic='sum', bins=[rbins,abins])[0])
            result[:,:,0]+=np.array(binned_statistic_2d(newr,aph,dEph,statistic='sum', bins=[rbins,abins])[0])
            asum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,aph,statistic='sum', bins=[rbins,abins])[0])
            zsum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,zaph,statistic='sum', bins=[rbins,abins])[0])
            rsum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,newr,statistic='sum', bins=[rbins,abins])[0])
            Esum[:,:,0]+=np.array(binned_statistic_2d(newr[1:],aph[1:],newE,statistic='sum', bins=[rbins,abins])[0])
            result[:,:,1]+=np.array(binned_statistic_2d(newr,aph,dEph*dEph,statistic='sum', bins=[rbins,abins])[0])
            asum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,aph*aph,statistic='sum', bins=[rbins,abins])[0])
            zsum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,zaph*zaph,statistic='sum', bins=[rbins,abins])[0])
            rsum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,newr*newr,statistic='sum', bins=[rbins,abins])[0])
            Esum[:,:,1]+=np.array(binned_statistic_2d(newr[1:],aph[1:],newE*newE,statistic='sum', bins=[rbins,abins])[0])
            
            # masterx.append(np.array(xlist)/mpc)
            # mastery.append(np.array(ylist)/mpc)
            # masterz.append(np.array(zlist)/mpc)
            # mastera.append(np.array(alist))
            # masterE.append(np.array(Elist))
            if (rank == 0) & (not i % 100):
                print(i,'of ',N)
            #if not warning:    
                #ax.plot(np.array(xlist)/mpc,np.array(ylist)/mpc,marker='o',markersize=2,lw=1,color=green)
        # resRlist=None
        # resafarr=None
        # resNarr=None
        # if not newrank:
        #     resRlist=np.zeros([divy,N])
        #     resafarr=np.zeros([divy,N])
        #     resNarr=np.zeros([divy,N], dtype=int)
        # # if rank % divy:
        # #     Rlisttmp=np.array(Rlist)
        # #     comm.Send(Rlisttmp, dest=rank-1, tag=77)
        # # elif:
        # #     comm.Recv(Rlisttmp, source=rank+1, tag=77)
        # #     resRlist[:,:]=[Rlist,Rlisttmp]
        
        # newcomm.Gather(np.array(Rlist), resRlist, root=0)
        # newcomm.Gather(np.array(afarr), resafarr, root=0)
        # newcomm.Gather(np.array(Narr), resNarr, root=0)
        
        photcount=comm.gather(photcount,root=0)
        result=comm.gather(result,root=0)
        asum=comm.gather(asum,root=0)
        zsum=comm.gather(zsum,root=0)
        rsum=comm.gather(rsum,root=0)
        Esum=comm.gather(Esum,root=0)
        # masterx=newcomm.gather(masterx,root=0)
        # mastery=newcomm.gather(mastery,root=0)
        # masterz=newcomm.gather(masterz,root=0)
        # mastera=newcomm.gather(mastera,root=0)
        # masterE=newcomm.gather(masterE,root=0)
        
        if not rank:
            # print(rank,Eim)
            # Rlist=resRlist.flatten()
            # afarr=resafarr.flatten()
            # Narr=resNarr.flatten()
            # masterx=[np.asarray(item,dtype=float) for sublist in masterx for item in sublist]
            # mastery=[np.asarray(item,dtype=float) for sublist in mastery for item in sublist]
            # masterz=[np.asarray(item,dtype=float) for sublist in masterz for item in sublist]
            # mastera=[np.asarray(item,dtype=float) for sublist in mastera for item in sublist]
            # masterE=[np.asarray(item,dtype=float) for sublist in masterE for item in sublist]
            mphotcount=np.zeros(np.shape(photcount[0]))
            mresult=np.zeros(np.shape(photcount[0]))
            masum=np.zeros(np.shape(photcount[0]))
            mzsum=np.zeros(np.shape(photcount[0]))
            mrsum=np.zeros(np.shape(photcount[0]))
            mEsum=np.zeros(np.shape(photcount[0]))
            for ss in range(len(photcount)):
                mphotcount+=photcount[ss]
                mresult+=result[ss]
                masum+=asum[ss]
                mzsum+=zsum[ss]
                mrsum+=rsum[ss]
                mEsum+=Esum[ss]
            tit='z'+str(int(1/astart-1))+'_'+'E'+str(Eim[ee])+'_'+str(Ntot)+'binned'
            fil=tit+'.pkl'
            tit=tit+'.pdf'
            with open('./pickle/'+fil, "wb") as f:
                pickle.dump(mphotcount,f)
                pickle.dump(mresult,f)
                pickle.dump(masum,f)
                pickle.dump(mzsum,f)
                pickle.dump(mrsum,f)
                pickle.dump(mEsum,f)
            #with open(fil,"rb") as f:
            #Rlist=pickle.load(f)
            #afarr=pickle.load(f)
            #ax.set_xlabel('x (Mpc)')
            #ax.set_ylabel('y (Mpc)')
            #ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            #plt.savefig('2drad'+tit)    
            #plt.show()
            # fig,ax=plt.subplots(1,1)
            # ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            # ax.set_xlabel('r (Mpc)')
            # ax.set_ylabel('Percent of Photons (N='+str(Ntot)+')')
            # weights=np.ones_like(Rlist)/float(len(Rlist))
            # ax.hist(np.array(Rlist), bins=50, weights=weights)
            # plt.savefig('rhist'+tit)
            # plt.cla()
            # ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            # ax.set_xlabel('final z')
            # ax.set_ylabel('Percent of Photons (N='+str(Ntot)+')')
            # weights=np.ones_like(afarr)/float(len(afarr))
            # ax.hist(np.array(afarr), bins=50, weights=weights)
            # plt.savefig('zhist'+tit)
            # plt.cla()
            # ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            # ax.set_xlabel('r (Mpc)')
            # ax.set_ylabel('E/Ei')
            # r=np.sqrt(np.array(xlist)**2+np.array(ylist)**2+np.array(zlist)**2)/mpc
            # ax.plot(r[:-1],np.array(Elist)/Ei,marker='o',markersize=2,lw=2,color=orange)
            # plt.savefig('r'+str(int((1-thresh)*100))+'_vs_E'+str(Eim[ee])+'phot.pdf')
            # plt.cla()
            
