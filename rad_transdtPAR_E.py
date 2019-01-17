from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle
from mpi4py import MPI

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

def rotmat(thet,phi):
    return [[np.cos(phi)*np.cos(thet),-np.sin(phi)*np.cos(thet),np.sin(thet)],
            [np.sin(phi),np.cos(phi),0],
            [np.sin(thet)*np.cos(phi),np.sin(phi)*np.sin(thet),np.cos(thet)]]
    


#if __name__ =="__main__":



#CONSTANTS
me=.511*10**6
alist=[1/1401.,1/1301.,1/1201]

zfinal=50.
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
#CONSTANTS
    
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

tt=[0.001] #Thres
Eilist=[0.1,1,10]
divy=int(size/len(Eilist))
if size == 1:
    divy=1
surplus=size-(divy*len(Eilist))
if size != 1:
    assert not surplus
Ntot=5000
N=int(Ntot / divy)
Eim=[Eilist[rank % len(Eilist)]]
color= rank % len(Eilist)
newcomm=comm.Split(color,rank)
newrank=newcomm.Get_rank()
comm.Barrier()

for astart in alist:
    for ee in range(len(Eim)):
        Ei=Eim[ee]*me
        afarr=np.zeros(N)
        Rlist=np.zeros(N)
        Narr=np.zeros(N, dtype=int)
        masterR=[]
        mastera=[]
        masterE=[]
        for i in range(N):
            phistep=np.random.uniform(0,2*pi)
            thetastep=np.random.uniform(0,pi)
            xlist=[0]
            ylist=[0]
            zlist=[0]
            
            dlna=0.0
            Elist=[Ei]
            alist=[astart]
            Eistep=np.copy(Ei)
            Efstep=np.copy(Ei)
            count=0
            ai=np.copy(astart)
            warning=0
            rot=1.
            while 1/ai-1 > zfinal:
                astep=np.copy(ai)
                ptmp=nh0/(astep**(3/2.)*np.sqrt(omegaM)*H0)*c*cross(Eistep)
                dlna=np.min([0.001/ptmp,0.001])
                pxstep=ptmp*dlna
                ppull=np.random.uniform(0,1)
                while pxstep < ppull:
                    ppull=np.random.uniform(0,1)
                    Eistep=Eistep*astep/(astep*(1+dlna))
                    astep=astep*(1+dlna)
                    if 1/astep-1 < zfinal:
                        break
                    ptmp=nh0/(astep**(3/2.)*np.sqrt(omegaM)*H0)*c*cross(Eistep)
                    dlna=np.min([0.001/ptmp,0.001])
                    pxstep=ptmp*dlna
                if 1/astep-1 < zfinal:
                    warning = 1
                    #ax.plot(np.array(xlist)/mpc,np.array(ylist)/mpc,marker='^',markersize=2,lw=1,color=red)
                    break
                Lstep=(2/(H0*np.sqrt(omegaM))*(astep**(1/2.)-ai**(1/2.)))*c    
                vec=[[np.sin(thetastep)*np.cos(phistep)*Lstep],
                     [np.sin(thetastep)*np.sin(phistep)*Lstep],
                     [np.cos(thetastep)*Lstep]]
                #this vector is v.Rz_1.Ry_1.Rz_2.Ry_2...->v.rot Thus I need to apply the reverse to get v, the desired vector in my original coordinate system
                xlist.append(np.dot(rot,vec)[0]+xlist[-1])
                ylist.append(np.dot(rot,vec)[1]+ylist[-1])
                zlist.append(np.dot(rot,vec)[2]+zlist[-1])

                phistep=np.random.uniform(0,2*pi)
                thetastep=np.arccos(rej(-1,1,pdfthet,maxPT,Eistep))
                Efstep=1/(2/me*np.sin(thetastep/2.)**2+1/Eistep)
                if Efstep > Eistep:
                    pdb.set_trace()
                
                rot=np.dot(rot,rotmat(thetastep,phistep))
                Eistep=Efstep#*ai/astep
                ai=np.copy(astep)
                Elist.append(Eistep)
                alist.append(astep)
                count+=1
                

            afarr[i]=1/ai-1
            Narr[i]=count
            Rarr=np.sqrt(np.array(xlist)**2+np.array(ylist)**2+np.array(zlist)**2)/mpc
            masterR.append(np.array(Rarr))
            mastera.append(np.array(alist))
            masterE.append(np.array(Elist))
            Rlist[i]=Rarr[-1]
            if rank == 0:
                print(i,'of ',N)
            #if not warning:    
                #ax.plot(np.array(xlist)/mpc,np.array(ylist)/mpc,marker='o',markersize=2,lw=1,color=green)
        resRlist=None
        resafarr=None
        resNarr=None
        if not newrank:
            resRlist=np.zeros([divy,N])
            resafarr=np.zeros([divy,N])
            resNarr=np.zeros([divy,N], dtype=int)
        # if rank % divy:
        #     Rlisttmp=np.array(Rlist)
        #     comm.Send(Rlisttmp, dest=rank-1, tag=77)
        # elif:
        #     comm.Recv(Rlisttmp, source=rank+1, tag=77)
        #     resRlist[:,:]=[Rlist,Rlisttmp]
        
        newcomm.Gather(np.array(Rlist), resRlist, root=0)
        newcomm.Gather(np.array(afarr), resafarr, root=0)
        newcomm.Gather(np.array(Narr), resNarr, root=0)
        masterR=newcomm.gather(masterR,root=0)
        mastera=newcomm.gather(mastera,root=0)
        masterE=newcomm.gather(masterE,root=0)
        
        if not newrank:
            print(rank,Eim)
            fig,ax=plt.subplots(1,1)
            Rlist=resRlist.flatten()
            afarr=resafarr.flatten()
            Narr=resNarr.flatten()
            masterR=[item for sublist in masterR for item in sublist]
            mastera=[item for sublist in mastera for item in sublist]
            masterE=[item for sublist in masterE for item in sublist]
            tit='par'+str(int((1-thresh)*100))+'E'+str(Eim[ee])+'_'+str(Ntot)
            fil=tit+'.pkl'
            tit=tit+'.pdf'
            with open(fil, "wb") as f:
                pickle.dump(np.array(Rlist),f)
                pickle.dump(np.array(afarr),f)
                pickle.dump(np.array(Narr),f)
            with open('full'+fil, "wb") as f:
                pickle.dump(masterR,f)
                pickle.dump(mastera,f)
                pickle.dump(masterE,f)
            #with open(fil,"rb") as f:
            #Rlist=pickle.load(f)
            #afarr=pickle.load(f)
            #ax.set_xlabel('x (Mpc)')
            #ax.set_ylabel('y (Mpc)')
            #ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            #plt.savefig('2drad'+tit)    
            #plt.show()
            ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            ax.set_xlabel('r (Mpc)')
            ax.set_ylabel('Percent of Photons (N='+str(Ntot)+')')
            weights=np.ones_like(Rlist)/float(len(Rlist))
            ax.hist(np.array(Rlist), bins=50, weights=weights)
            plt.savefig('rhist'+tit)
            plt.cla()
            ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            ax.set_xlabel('final z')
            ax.set_ylabel('Percent of Photons (N='+str(Ntot)+')')
            weights=np.ones_like(afarr)/float(len(afarr))
            ax.hist(np.array(afarr), bins=50, weights=weights)
            plt.savefig('zhist'+tit)
            plt.cla()
            # ax.set_title('Ei='+str(Eim[ee])+'*M_e')
            # ax.set_xlabel('r (Mpc)')
            # ax.set_ylabel('E/Ei')
            # r=np.sqrt(np.array(xlist)**2+np.array(ylist)**2+np.array(zlist)**2)/mpc
            # ax.plot(r[:-1],np.array(Elist)/Ei,marker='o',markersize=2,lw=2,color=orange)
            # plt.savefig('r'+str(int((1-thresh)*100))+'_vs_E'+str(Eim[ee])+'phot.pdf')
            # plt.cla()
            
