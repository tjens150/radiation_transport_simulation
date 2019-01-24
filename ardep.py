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
Eread=[0.1,1,10] #m_e
Emax=max(Eread) #m_e
Emin=0.05 #m_e
zread=[1200,1300,1400]
thresh=100 #at least this many photons given an astep bin

astart=1/1401.
afinal=1/51.
astep=0.1
nEbins=3
rstep=0.1 #mpc
rmax=200 #mpc
alogbins=np.arange(np.log(astart),np.log(afinal),astep)
Ebins=np.linspace(me*Emax,me*Emin,nEbins)
Ebins=[10*me,5*me, 1.1*me,0.11*me,0.05*me]
rbins=np.arange(0,rmax,rstep)
nrbins=int(rmax/0.1)

x=[]
y=[]
z=[]
a=[]
E=[]
dE=[]
num=[]
# xdic={}
# ydic={}
# zdic={}
# adic={}
# Edic={}
count=0
print('Loading in data...')
numdata=5000
for zi in zread:
    for Ei in Eread:
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
        reparr=[len(item) for item in mastera]
        num.append(np.repeat(np.arange(numdata)+count*numdata,reparr))
        x.append(np.concatenate(masterx))
        y.append(np.concatenate(mastery))
        z.append(np.concatenate(masterz))
        a.append(np.concatenate(mastera))
        E.append(np.concatenate(masterE))
        dE.append(np.concatenate(tmpdE))
        count+=1
        # xdic['z%s_E%s' % (zi,Ei)]=masterx
        # ydic['z%s_E%s' % (zi,Ei)]=mastery
        # zdic['z%s_E%s' % (zi,Ei)]=masterz
        # adic['z%s_E%s' % (zi,Ei)]=mastera
        # Edic['z%s_E%s' % (zi,Ei)]=masterE
x=np.concatenate(x)
y=np.concatenate(y)
z=np.concatenate(z)
a=np.concatenate(a)
E=np.concatenate(E)
dE=np.concatenate(dE)
num=np.concatenate(num)

print('Done loading, starting binning')

assign_a=np.digitize(np.log(a),alogbins)
totnum=0.
result=np.zeros([nrbins,len(alogbins)])
rdep={}
#nphotind={}
nphotbin={}
amean={}
Emean={}
ttot=time.time()

for i in reversed(range(1,len(alogbins))):
    wh_a=(assign_a == i)
    wh_ale=(assign_a >= i)
    assign_E=np.digitize(E[wh_a],Ebins,right=True)
    # if not np.all(assign2 == len(Ebins)):
    for j in range(1,len(Ebins)):
        dicstr='z%s_E%s' % (int(1/np.exp(alogbins[i])-1),Ebins[j]/me)
        wh_E=(assign_E == j)
        uid,uind=np.unique(num[wh_a][wh_E], return_index=True)
        numphot=uid.size
        totnum+=numphot
        tbin=time.time()
        numale=num[wh_ale]
        xale=x[wh_ale]
        yale=y[wh_ale]
        zale=z[wh_ale]
        dEale=dE[wh_ale]
        aale=np.log(a[wh_ale])
        if numphot > thresh:
            print('Number of photons: %s' % (numphot))
            for k in uid:
                #tmptime=time.time()
                wh_ph=(numale == k)
                newx=xale[wh_ph]-xale[wh_ph][0]
                newy=yale[wh_ph]-yale[wh_ph][0]
                newz=zale[wh_ph]-zale[wh_ph][0]
                newr=np.sqrt(newx*newx+newy*newy+newz*newz)
                lim=newr < rmax
                dEph=dEale[wh_ph][lim]
                aph=aale[wh_ph][lim]
                dEph[0]=0.
                result+=binned_statistic_2d(newr[lim],aph,dEph,statistic='sum', bins=[rbins,alogbins])
                #result+=np.array(binned_statistic(newr[lim], dEph, np.sum, nrbins, [0,rmax]))
                #assign_r=np.digitize(newr,rbins)
                #result+=np.array([dE[assign_r == tt].sum() for tt in range(1, len(rbins))])
                #print(time.time() - tmptime)
            rdep[dicstr]=result
            nphotbin[dicstr]=numphot
            amean[dicstr]=a[wh_a][wh_E][uind].mean()
            Emean[dicstr]=E[wh_a][wh_E][uind].mean()
            print('i=%s, j=%s bin takes %s' % (i,j,time.time()-tbin))
                #/(astep*4/3*pi*(rbins[tt-1]**3-rbins[tt]**3))
with open('rdep45000_v1.pkl', "wb") as f:
    pickle.dump(rdep,f)
    pickle.dump(totnum,f)
    pickle.dump(nphotbin,f)
    #pickle.dump(nphotind,f)
    pickle.dump(amean,f)
    pickle.dump(Emean,f)
    
print(time.time()-ttot)
