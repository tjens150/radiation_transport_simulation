from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle
from mpi4py import MPI
import time
from scipy.sparse import csr_matrix
from scipy.stats import binned_statistic_2d

def binned_statistic(xx, values, func, nbins, rrange):
    
    'The usage is nearly the same as scipy.stats.binned_statistic'
    
    N = len(values)
    r0, r1 = rrange
    
    digitized = (float(nbins)/(r1 - r0)*(xx - r0)).astype(int)

    S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))
    
    return [func(group) for group in np.split(S.data, S.indptr[1:-1])]


#CONSTANTS
me=.511*10**6
c=299792458.0
mpc=3.086*10**22
sigT=6.652459*10**(-29.)
nh0=8.6*0.022
H0=2.197*10**(-18)
omegaM=0.308
#CONSTANTS


fol='/Users/Acolyte/NYU/Research/rad_trans/full5000Photons/'
Eread1=[5] #m_e
Eread2=[0.1]
Emax=max(Eread1) #m_e
Emin=0.05 #m_e
zread1=[1300,1250,1150,1000,900,800,700,600,500,400]
zread2=[1300,1200,900,800,700,600,500,400]
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
            
# x=np.concatenate(x)
# y=np.concatenate(y)
# z=np.concatenate(z)
# a=np.concatenate(a)
# E=np.concatenate(E)
# dE=np.concatenate(dE)
# num=np.concatenate(num)

print('Done loading, starting binning')

totnum=0.
rdep={}
adist={}
rdist={}
photcountdic={}
#nphotind={}
nphotbin={}
astat={}
Estat={}
ttot=time.time()



for key in xdic:
    tbin=time.time()
    result=np.zeros([len(rbins)-1,len(alogbins)-1,2])
    photcount=np.zeros([len(rbins)-1,len(alogbins)-1])
    asum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
    rsum=np.zeros([len(rbins)-1,len(alogbins)-1,2])
    for k in range(len(xdic[key])):
        newr=rdic[key][k]
        aph=np.log(adic[key][k])
        zaph=1/adic[key][k]-1
        dEph=dEdic[key][k]
        #pdb.set_trace()
        photcount+=np.array(binned_statistic_2d(newr,aph,np.ones_like(newr,dtype='int'),statistic='sum', bins=[rbins,alogbins])[0])
        result[:,:,0]+=np.array(binned_statistic_2d(newr,aph,dEph,statistic='sum', bins=[rbins,alogbins])[0])
        asum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,zaph,statistic='sum', bins=[rbins,alogbins])[0])
        rsum[:,:,0]+=np.array(binned_statistic_2d(newr,aph,newr,statistic='sum', bins=[rbins,alogbins])[0])
        result[:,:,1]+=np.array(binned_statistic_2d(newr,aph,dEph*dEph,statistic='sum', bins=[rbins,alogbins])[0])
        asum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,zaph*zaph,statistic='sum', bins=[rbins,alogbins])[0])
        rsum[:,:,1]+=np.array(binned_statistic_2d(newr,aph,newr*newr,statistic='sum', bins=[rbins,alogbins])[0])
        
    rdep[key]=result
    adist[key]=asum
    rdist[key]=rsum
    photcountdic[key]=photcount
    nphotbin[key]=len(xdic[key])
    astat[key]=np.array([zEdic[key][0],0])
    Estat[key]=np.array([zEdic[key][1],0])
    print(key+' bin takes %s' % (time.time()-tbin))

    #/(astep*4/3*pi*(rbins[tt-1]**3-rbins[tt]**3))
with open(fol+'/comp/ardep_comp_v2.pkl', "wb") as f:
    pickle.dump(rdep,f)
    pickle.dump(adist,f)
    pickle.dump(rdist,f)
    pickle.dump(photcountdic,f)
    pickle.dump(nphotbin,f)
    #pickle.dump(nphotind,f)
    pickle.dump(astat,f)
    pickle.dump(Estat,f)
    
print(time.time()-ttot)
