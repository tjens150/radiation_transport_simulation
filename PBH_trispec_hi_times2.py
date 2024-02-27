#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
from math import pi
import pickle
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import time
import array
from collections import OrderedDict
from scipy.special import spherical_jn
from classy import Class
import _wignerpy as wigpy
#from sympy.physics.wigner import wigner_3j
#import py3nj
from _gen_Q_geo_cy import gen_Qsq, gen_QQt, gen_Qtsq, gen_Qt_trans, gen_Qt_scat
from _gen_C_ell_cy import gen_C_ellx4_slow
from mpi4py import MPI
plt.rcParams['figure.figsize']=[12,8]


blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)


#CONSTANTS all in EV, meters, seconds
me=.511*10**6 #electron mass
c=299792458.0 #speed of light
mpc=3.086*10**22 #Mpc in m
zfinal=50. #end of simulation
sigT=6.652459*10**(-29.) #thomson cross section

#PLANCK 2018
#Cosmological parameters PLANCK 2018
h=0.6736#0.67
YHe=0.241#0.24 #fraction of Helium from BBN
H0=100.*h*1000./mpc # Hubble constant in seconds
T0=2.725#2.73 #CMB temp today
Nnueff=3.046 #dof of Neutrinos
omegaM=0.1430/h**2 #matter density
omegaR=T0**4*4.48162687719e-7*(1+0.227107318*Nnueff)/h**2 #relativistic density
omegaK=0. #Curvature density
omegaDE=1-omegaM-omegaR-omegaK #Dark Energy density
aeq=4.15e-5/(omegaM*h**2) #matter radiation equaltiy
omegab=0.02237/h**2#0.02237/h**2 #baryon density
nh0=(1-YHe)*11.3*omegab*h**2 #density of hydrogen today
alpha=1/137.
hbar=6.582e-16 #eV s
kb=8.61733e-5 #eV K^-1
radconst=4723. #eV/m^3 K^4 8*np.pi**5*kb**4*(1/(c*hbar*2*np.pi))**3/15.
rnot=np.sqrt(sigT*3/(8*np.pi))
Heco=YHe/(4.*(1.-YHe)) #Helium number fraction
EI=13.6 #Ionization energy of neutral hydrogen
As=np.exp(3.044)*1e-10#np.exp(3.044)*1e-10
ns=0.9649#0.966
kpivot=0.05 #Mpc^-1
tau_reio=0.0544

#PLANCK 2018
#CONSTANTS

#get_ipython().magic('load_ext Cython')



def trap_int(quad,x,axis=0):
    q1=quad.take(indices=list(range(1, quad.shape[axis])), axis=axis)
    q2=quad.take(indices=list(range(0,quad.shape[axis]-1)),axis=axis)
    qint=q1+q2
    shape = np.swapaxes(qint, qint.ndim-1, axis).shape
    dx=np.swapaxes(np.broadcast_to(x[1:]-x[:-1],shape), qint.ndim-1, axis)
    return (qint*dx).sum(axis=axis)/2.


def HubbleRate(a):
    return H0*np.sqrt(omegaM/a**3+omegaK/a**2+omegaDE+omegaR/a**4)


def find_nearest(array, value):
    array = np.asarray(array)
    array[~np.isfinite(array)]=0.
    idx = (np.abs(array - value)).argmin()
    return idx

def conformal(a):
    N=1000
    if type(a) != np.ndarray: a=np.asarray([a])
    if a.shape == (): a=np.asarray([a])
    it=a.size
    val=np.zeros(it)
    for i in range(it):
        aarr=np.exp(np.linspace(np.log(a[i]),np.log(1),N))
        da=(0-np.log(a[i]))/np.float(N)
        quad=1/(HubbleRate(aarr)*aarr)
        val[i]=c/mpc*(quad[1:]+quad[:-1]).sum()*da/2.
    return val

def plot_an(x, y, z, zplts, xlim=None,xscale='log',legloc=None, ylog=0,savfol=None,ylim=None,xlabel='x',ylabel='y',tit='title',pdf=None,legendbool=True,pltmarker=None,fig=None,ax=None,linestyle='-',legstr=r'$z$='):
    plt.rcParams['figure.figsize']=[12,8]
    def find_nearest(array, value):
        array = np.asarray(array)
        array[~np.isfinite(array)]=0.
        idx = (np.abs(array - value)).argmin()
        return idx
    #Colors for plot
    import colormaps as cmaps
    plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    cmapV=plt.cm.get_cmap('viridis')
    rgord=np.linspace(0,1,len(zplts))
    rgba=cmapV(rgord)
    prevind=-1
    if ax == None: fig, ax=plt.subplots(1,1)
    for cc in range(len(zplts)):
        indx=find_nearest(z,zplts[cc]) #find the bin closest to desired z-plot
        if indx == prevind: continue
        if indx == len(y[0,:]):indx-=1
        prevind = indx
        ax.plot(x,y[:,indx],lw=2,linestyle=linestyle, color=rgba[cc], marker=pltmarker, zorder=cc,label=legstr+'%s' % (z[indx]) )#/Gnorm[indx]

    if legendbool:
        ll=ax.legend(fontsize=14,loc=legloc,ncol=2,labelspacing=0.2,handlelength=0.75)
        ll.set_zorder(10000)
    ax.set_xlabel(xlabel,fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.set_title(r'$E_{\rm inj}=$%s $(m_e)$, $z_{\rm inj}=$%s, $N_{\rm tot}=$%s' % (int(Eparam*1000)/1000., int(1/ai-1), Ntot), fontsize=18)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_title(tit, fontsize=18)
    if xscale == 'log':
        ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    fig.tight_layout()
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if ax == None:
        if (savfol) or (pdf) == None:
            plt.show()
        elif pdf == None:
            plt.savefig(savfol+'z%s_E%s_N%s.pdf' % (int(1/ai-1),int(Eparam),int(Ntot)))
            plt.close(fig)
        else:
            plt.savefig(pdf)
            plt.close()



#This is a general secant-differentiation function. It returns the derivative of a f(x) with the same resolution as the input x.
def sec_diff(x,y,axis=0): 
    reshapearr=np.insert(np.ones(y.ndim-1,dtype=int),axis,int(x[2:].size))
    if axis==0:
        diff=(y[2:,...]-y[:-2,...])/(x[2:]-x[:-2]).reshape(reshapearr)
        diff=np.insert(np.insert(diff,0,diff[0,...],axis=axis),diff.shape[0]+1,diff[-1,...],axis=axis)
        return diff
    if axis==1:
        diff=(y[:,2:,...]-y[:,:-2,...])/(x[2:]-x[:-2]).reshape(reshapearr)
        diff=np.insert(np.insert(diff,0,diff[:,0,...],axis=axis),diff.shape[axis]+1,diff[:,-1,...],axis=axis)
        return diff
    if axis==2:
        diff=(y[:,:,2:,...]-y[:,:,:-2,...])/(x[2:]-x[:-2]).reshape(reshapearr)
        diff=np.insert(np.insert(diff,0,diff[:,:,0,...],axis=axis),diff.shape[axis]+1,diff[:,:,-1,...],axis=axis)
        return diff
    
#This generates the source function used in the line of sight multiplied by the visibility function.
def gen_S0(s,k,tau,isw=True):
    S0=np.zeros([3,tau.size,k.size])
    if isw: #include integrated saches-wolfe effect
        first=s['delta_g']/4.+s['psi']+sec_diff(tau,s['psi']+s['phi'],axis=1)/kappadot_int(tau)
    else:
        first=s['delta_g']/4.+s['psi']
    second=s['theta_b']/k[:,np.newaxis]
    third=(s['shear_g']*2+s['mult_Pg_0']+s['mult_Pg_2'])/8.
    S0[0,:,:]=np.swapaxes(first+third/2.,0,1) #This term multiplies the zeroth legendre polynomial bessel
    S0[1,:,:]=np.swapaxes(second,0,1) #multiplies the first legendre polynomial bessel 
    S0[2,:,:]=np.swapaxes(3*third/2.,0,1) #multiplies the second legendre polynomial bessel
    return S0*g_int(tau)[np.newaxis,:,np.newaxis]

def gen_bessel_ell(l, x): #Generate the bessel function,first and second derivative
    xzero=(x==0.)
    jn=spherical_jn(l,x)
    jnp=spherical_jn(l,x,derivative=True)
    if l != 0:
        jnpp=spherical_jn(l-1,x,derivative=True)-(l+1)/(x)*jnp+(l+1)/(x)**2*jn
    else: jnpp=-spherical_jn(l+1,x,derivative=True)+(l/x)*jnp-l/x**2*jn
    if np.any(xzero):
        jnpp[xzero]=0.
    return jn, jnp, jnpp

def single_l(eta,x,unintsrc,l): #Generate each multipole as a function of k by integrating over chi
    jn,jnp,jnpp=gen_bessel_ell(l,x)
    if x.ndim ==1:
        bessel=[jn[:,np.newaxis],jnp[:,np.newaxis],jnpp[:,np.newaxis]]
    else: 
        bessel=[jn,jnp,jnpp]
    src=(unintsrc[0,:,:]*bessel[0]+unintsrc[1,:,:]*bessel[1]+unintsrc[2,:,:]*bessel[2])#*coeff
    #pdb.set_trace()
    if eta.ndim==1:
        return (((src[1:,:]+src[:-1,:])*np.abs(eta[1:,np.newaxis]-eta[:-1,np.newaxis])).sum(axis=0)/2.).squeeze()
    return (((src[1:,:]+src[:-1,:])*np.abs(eta[1:,:]-eta[:-1,:])).sum(axis=0)/2.).squeeze()

def S0_interp_2d(S0,k,tau,kint,tauint): #given that we integrate over x=chi*k, we have to interpolate to get the correct chi
    #S0_int=[]
    S0_ret=np.zeros([3,tauint.shape[0],kint.size])
    for j in range(3):
        S0_int=(interp2d(k,tau,S0[j,:,:]))#,bounds_error=False,fill_value=0.))
        for i in range(kint.size):
            kcut=(kint[i]<=k.max())*(kint[i]>=k.min())
            if kcut:
                taucut=(tauint[:,i]<=tau.max())*(tauint[:,i]>=tau.min())
                S0_ret[j,taucut,i]=np.flip(S0_int(kint[i],tauint[taucut,i]).squeeze())
        print(j)
     
    return S0_ret


def load_tab(alistiter,abins, rbins,fol,PI=0,suffix=None,dlnanam=0.01):
    znam=np.asarray(1/alistiter-1, dtype=int)

    ardep=np.zeros((rbins.size-1,abins.size-1,alistiter.size))
    nphotbin=np.copy(ardep)
    adist=np.copy(ardep)
    Etot=np.zeros(alistiter.size)
    #injstat=OrderedDict()

    #fol='/Users/Acolyte/NYU/Research/rad_trans/repo/pickle/photion/tab/'

    #znam=znam #redshifts below this caused problems with photons never scattering
    
    Eparam=0.447
    Nnam=1000000
    #dlnanam=0.01
    count=0
    if PI:
        print('Photoionized case')
        Eparam=20.1
    if suffix == None:
        suffix='.pkl'
    for zz in znam:
        fil='z%s_E_flat%s_N%s_dlna%s_%s' % (zz,Eparam,Nnam,dlnanam,suffix)
        zcut=abins[:-1] >= 1/1505.
        if zz > 1519:
            zcut=abins[:-1]>0
            fil='z%s_E_flat%s_N%s_dlna%s_tab_frac_bin_extend.pkl' % (zz,Eparam,Nnam,dlnanam)
        #fil2='pickle/mbh10_rmin1_ainjstep0.1/z%s_MBH%s_N%s_binned_Etot.pkl' % (zz,mbh,Nnam)
        with open(fol+fil,"rb") as f:
            ardep[:,zcut,count]=pickle.load(f, encoding="latin1")[:,:] #excise the last rbin because it's STUPID
            nphotbin[:,zcut,count]=pickle.load(f, encoding="latin1")[:,:]
            adist[:,zcut,count]=pickle.load(f, encoding="latin1")[:,:]
            Etot[count]=pickle.load(f, encoding="latin1")
            count+=1
    return ardep, nphotbin, adist, Etot


# Here we multiply our Green's function by a (a_inj/a_dep)^3/H_inj such that we input a rho_inj (not eps_inj) and interate over dlna_dep
def gen_greens(alistiter,abins,ardep,nphotbin,adist,Etot,karr): #generate G_dep^inj
    def dtfunc(a1,a2): #Compute the time between two scale factors in a matter+rad dom
        return 2/(3*H0*np.sqrt(omegaM))*(a2*np.sqrt(a2+aeq)-a1*np.sqrt(a1+aeq)+2*aeq*(np.sqrt(a1+aeq)-np.sqrt(a2+aeq)))
    
    #G=np.zeros_like(nphotbin)
    #cut=nphotbin != 0
#     amean=(adist.sum(axis=0))/nphotbin.sum(axis=0)
    amean=np.exp(np.log(abins[1:]*abins[:-1])/2.)[:,np.newaxis]
    dR=4*pi/3*(rbins[1:]**3-rbins[:-1]**3) #volume of each bin
    dt=(dtfunc(abins[:-1],abins[1:]))[:,np.newaxis]*(amean)**3/(alistiter[np.newaxis,:])**3 #time of each bin
    denom=dR[:,np.newaxis,np.newaxis]*dt[np.newaxis,:,:]
    tmpG=ardep/(Etot*HubbleRate(alistiter))[np.newaxis,np.newaxis,:]
    #G[cut]=(tmpG/denom)[cut]
    G=(tmpG/denom)
    Gnorm=(G*dR[:,np.newaxis,np.newaxis]).sum(axis=0)
    #slat=slattemp(amean,np.repeat(alistiter[np.newaxis,:],amean.shape[0],axis=0))
    def sincint(karr,r):
        r1=r[:-1,np.newaxis]
        r2=r[1:,np.newaxis]
        k=karr[np.newaxis,:]
        return (k*r1*np.cos(k*r1)-np.sin(k*r1)-k*r2*np.cos(k*r2)+np.sin(k*r2))/k**3
    #return 4*pi*(k*r1*(-6 + k**2*r1**2)*np.cos(k*r1) + k*r2*(6 - k**2*r2**2)*np.cos(k*r2) - 3*(-2 + k**2*r1**2)*np.sin(k*r1) + 3*(-2 + k**2*r2**2)*np.sin(k*r2))/k**5
    print('Generating k-space Greens function...')
    Gkquad=(G/Gnorm[np.newaxis,:,:])
    #Gkquad[~cut]=0.
    Gkquad[~np.isfinite(Gkquad)]=0.
    Gk=(Gkquad[:,np.newaxis,:,:]*sincint(karr,rbins)[:,:,np.newaxis,np.newaxis]).sum(axis=0)*4*np.pi
    return Gnorm, Gk #(adep,ainj) (k,adep,ainj)

def gen_W(alog_inj,alog_dep,z,G_homo,G_xe,G_k): #Generate G_xe^inj
    z_inj=1/np.exp(alog_inj)-1
    z_dep=1/np.exp(alog_dep)-1
    dlnadep=np.abs(alog_dep[1]-alog_dep[0])
    dlnainj=np.abs(alog_inj[1]-alog_inj[0])
    #should be redundant, but enforces the limits of integration
    tol=1e-3
    for zz in range(len(z_inj)):
        G_homo[:,zz]=np.where(z_dep <= z_inj[zz]+tol, G_homo[:,zz], 0.)
    for zz in range(len(z_dep)):
        G_xe[zz,:]=np.where(z <= z_dep[zz]+tol, G_xe[zz,:], 0.) 
    W_coeff=G_homo[np.newaxis,:,:]*G_xe.T[:,:,np.newaxis] #(:,a_dep,a_inj).(z,adep,:)
    W=np.zeros([z.size,alog_inj.size,G_k.shape[0]])
    for i in range(G_k.shape[0]):
        W_quad=W_coeff*G_k[np.newaxis,i,:,:]
        W[:,:,i]=((W_quad[:,1:-1,:]).sum(axis=1)+(W_quad[:,0,:]+W_quad[:,-1,:])/2.)*dlnadep #(z,a_inj)
    return W



def load_G_xe(Gtfil,numz=150):#Load in analytic ionization Green's function, this assumes it is integrated with rho_inj/(H n_H E_I), so we divide it out here instead of plug in \epsilon_inj/E_I
    ldf=np.loadtxt(Gtfil)
    G_t={}
    G_t['z_dep']=ldf[1:,0]
    G_t['z']=ldf[0,1:]
    G_t['G_t']=ldf[1:,1:]/(nh0/(1e6)*(1+ldf[1:,0])**3*HubbleRate(1/(1+ldf[1:,0]))*EI)[:,np.newaxis] #divide but an extra normalization
    res=np.int(G_t['z'].size/numz)
    if res == 0: res=1
    G_xe=G_t['G_t'][:,::res]
    z=G_t['z'][::res]
    return G_xe,z



def load_tcb(transf_file): #load in the transfer function of v_bc
    transf={}
    with open(transf_file,"rb") as f: #th== has one extra z_inj at the end
        transf['TF']=pickle.load(f, encoding="latin1")[:-1,:] #[z_inj,k]
        transf['k']=pickle.load(f, encoding="latin1")
        transf['z_inj']=pickle.load(f, encoding="latin1")[:-1]
    return transf
def vrms(Tarr,kint):
    quad=Tarr**2*As*((kint/kpivot)**(ns-1.)/kint)[np.newaxis,:]
    dk=kint[1:]-kint[:-1]
    return np.sqrt(((quad[:,1:]+quad[:,:-1])*dk[np.newaxis,:]).sum(axis=1)/2.) #in units of c


def load_inj(pbhmass,f=1,vsqvar=None,PI=False): #load in the energy injection data from modified HyRec
    suf_fol='./data/'
    if vsqvar == None:
        inj_file=suf_fol+'PBH_inj/Large_Scale/mbh%s_f%s_rho_rhovsq.dat' % (pbhmass,f)
        if PI: inj_file=suf_fol+'PBH_inj/Large_Scale/mbh%s_f%s_rho_rhovsq_PI.dat' % (pbhmass,f)
    else:
        if f==1:inj_file=suf_fol+'PBH_inj/Amplitude/PBH_MBH_%s.dat' % pbhmass
        else:inj_file=suf_fol+'PBH_inj/Amplitude/PBH_MBH_%s_f%s.dat' % (pbhmass,f)
        if PI: 
            if f==1: inj_file=suf_fol+'PBH_inj/Amplitude/PBH_MBH_%s_PI.dat' % pbhmass
            else: inj_file=suf_fol+'PBH_inj/Amplitude/PBH_MBH_%s_PI_f%s.dat' % (pbhmass,f)
    ldf=np.loadtxt(inj_file)
    inj={}
    inj['z']=ldf[:-1,0]
    inj['inj']=ldf[:-1,1]
    inj_int=interp1d(np.log(1/(1+inj['z'])),inj['inj'],fill_value=0.,kind='cubic')
    if vsqvar == None:
        inj['inj_v_sq']=ldf[:-1,2]/(c*100.)**2 #cm/s -> c units
        inj_sq_int=interp1d(np.log(1/(1+inj['z'])),inj['inj_v_sq'],fill_value=0.,kind='cubic')
    else:
        print('Using vsq variance')
        inj['inj_sq']=ldf[:-1,2]
        inj_sq_int=interp1d(np.log(1/(1+inj['z'])),inj['inj_sq'],fill_value=0.,kind='cubic')
    return inj_int, inj_sq_int

def gen_b(pbhmass,loga,vrmsval,f=1,vsqvar=None,PI=False): #Generate the bias parameter: b*\rho_inj/<v_bc^2>
    inj_int,inj_sq_int=load_inj(pbhmass,f=f,vsqvar=vsqvar,PI=PI)
    if vsqvar == None:
        b=3/2.*(inj_sq_int(loga)/(inj_int(loga)*vrmsval**2)-1)
        b=b*inj_int(loga)/vrmsval**2 #ensures large scale
    else:
        print('b from matching vsq var')
        b=-np.sqrt(3/2.*(inj_sq_int(loga)/inj_int(loga)**2-1))
        b=b*inj_int(loga)/vrmsval**2
    return b



def interp_G_e_new(G_e,z,karr,tau,kint): #Interpolate G_e(\sqrt{2}k)
    Del_e=np.zeros([z.size,kint.size])
    for i in np.arange(z.size):
        tmp_int=interp1d(karr,G_e[i,:],bounds_error=False,fill_value='extrapolate')
        Del_e[i,:]=tmp_int(np.sqrt(2)*kint)*transf_int(kint)/np.sqrt(G_e[i,0]) #this sqrt is the reason we took a minus sign in the first place
    if G_e[i,0] < 0.: print('warning')
    Del_e_fin=np.zeros([tau.size,kint.size])
    for i in np.arange(kint.size):
        tmp_int=interp1d(z,Del_e[:,i],bounds_error=False,fill_value='extrapolate') 
        Del_e_fin[:,i]=tmp_int(z_int(tau))
    return Del_e_fin




def gen_l_arr(l,l2_max): #Generate an \ell that explores every possible wig_3j symbol
    l1=np.asarray([0])
    for i in np.arange(l.size):
        l1=np.append(l1,(np.arange(l2_max*2+1)-l2_max+l[i]))
    return np.unique(l1[l1>= 0])

#gen wig_3j symbol, if spin=1, then gen g_{l1,l2,l3}^{1,-1,0}
def gen_wig3j(l1,l2,l3,thiswig=wigpy.wigner3j,spin=0):
    l1s=l1.size
    l2s=l2.size
    l3s=l3.size
    wig3j=np.zeros([l1s,l2s,l3s])
    for i in np.arange(l1s):
        for j in np.arange(l2s):
            for k in np.arange(l3s):
                if spin==0:
                    if ((l1[i]+l2[j]+l3[k]) % 2) ==0:
                        wig3j[i,j,k]=thiswig(l1[i],l2[j],l3[k],0,0,0)
                if spin==1:
                    wig3j[i,j,k]=thiswig(l1[i],l2[j],l3[k],1,-1,0)
        if not (i % np.floor(l1s/10)): print(('%s of %s' % (i, l1s)))
    return wig3j.squeeze()

# def gen_wig3j(l1t,l2t,l3t,thiswig=py3nj.wigner3j,spin=0):
#     l1=l1t*2
#     l2=l2t*2
#     l3=l3t*2
#     maxind=np.argmax([l1.size,l2.size,l3.size])
#     wig3j=np.zeros([l1.size,l2.size,l3.size])
#     if maxind==0:
#         for i in np.arange(l2.size):
#             for j in np.arange(l3.size):
#                 if spin==0:
#                     wig3j[:,i,j]=thiswig(l1,l2[i],l3[j],0,0,0)
#                 if spin==1:
#                     wig3j[:,i,j]=thiswig(l1,l2[i],l3[j],2,-2,0)
#             if not (i % np.floor(l2.size/10)): print(('%s of %s' % (i, l2.size)))
#     if maxind==1:
#         for i in np.arange(l1.size):
#             for j in np.arange(l3.size):
#                 if spin==0:
#                     wig3j[i,:,j]=thiswig(l1[i],l2,l3[j],0,0,0)
#                 if spin==1:
#                     wig3j[i,:,j]=thiswig(l1[i],l2,l3[j],2,-2,0)
#             if not (i % np.floor(l1.size/10)): print(('%s of %s' % (i, l1.size)))
#     if maxind==2:
#         for i in np.arange(l1.size):
#             for j in np.arange(l2.size):
#                 if spin==0:
#                     wig3j[i,j,:]=thiswig(l1[i],l2[j],l3,0,0,0)
#                 if spin==1:
#                     wig3j[i,j,:]=thiswig(l1[i],l2[j],l3,2,-2,0)
#             if not (i % np.floor(l1.size/10)): print(('%s of %s' % (i, l1.size)))
#     return wig3j.squeeze()

def gen_Cthet_arr(s,tau,k,k_trsf,l2,interpkind='linear'): #Gen C_thet array to some max \ell
    ret_T_1=np.zeros([l2.size,k_trsf.size,tau.size])
    for j in np.arange(l2.size):
        if l2[j] == 0:
            this_s=0.
        elif l2[j] == 1:
            this_s=((-s['theta_g']+s['theta_b'])/(3*k)[:,np.newaxis])
        elif l2[j] == 2:
            this_Pi=(s['shear_g']*2.+s['mult_Pg_0']+s['mult_Pg_2'])/8.
            this_s=(-s['shear_g']/2.+this_Pi/5.)
        else:
            this_s=(-s['mult_Tg_{}'.format(l2[j])]/4.)
        if l2[j] != 0:
            for i in range(tau.size):
                ret_T_1[j,:,i]=(interp1d(k,this_s[:,i],kind=interpkind))(k_trsf)
    return ret_T_1

def gen_Cthet_arr_hi(s,tau,tauhi,k,k_trsf,l2,interpkind='linear'): #Gen C_thet array to some max \ell
    ret_T_1=np.zeros([l2.size,k_trsf.size,tauhi.size])
    tmp=np.zeros([k.size,tauhi.size])
    for j in np.arange(l2.size):
        if l2[j] == 0:
            this_s=0.
        elif l2[j] == 1:
            this_s=((-s['theta_g']+s['theta_b'])/(3*k)[:,np.newaxis])
        elif l2[j] == 2:
            this_Pi=(s['shear_g']*2.+s['mult_Pg_0']+s['mult_Pg_2'])/8.
            this_s=(-s['shear_g']/2.+this_Pi/5.)
        else:
            this_s=(-s['mult_Tg_{}'.format(l2[j])]/4.)
        if l2[j] != 0:
            for i in range(k.size):
                tmp[i,:]=(interp1d(tau,this_s[i,:],kind=interpkind))(tauhi)
            for i in range(tauhi.size):
                ret_T_1[j,:,i]=(interp1d(k,tmp[:,i],kind=interpkind))(k_trsf)
    return ret_T_1

def gen_bess_fast(l,x,derivative=False): #gen bessel faster if we are populating entire every integer \ell
    global l_mark,jl,jl_n
    if l == 0:
        if l_mark != 1: print('ERROR, l_mark is wrong')
        return jl_n
    if l == 1:
        if l_mark != 1: print('ERROR, l_mark is wrong')
        return jl
    ret_jl=(jl)
    ret_jl_n=(jl_n)
    tmp=np.zeros_like(jl)
    cut=(x>l)
    excut=(~cut)
    for i in range(l_mark,l):
        tmp[cut]=(2*i+1)*ret_jl[cut]/x[cut]-ret_jl_n[cut]
        ret_jl_n=np.copy(ret_jl)
        ret_jl=np.copy(tmp)
    l_mark=l
    ret_jl[excut]=spherical_jn(l,x[excut])
    jl=(ret_jl)
    jl_n=(ret_jl_n)
    if derivative:
        jl_deriv=jl_n-(l+1)*jl/x
        jl_deriv[excut]=spherical_jn(l,x[excut],derivative=True)
        return ret_jl, jl_deriv
    return ret_jl


#function to generate J_ell
def gen_J_ell_py(T_1_arr,interparr,l1,l2,l3,wig3j=None): #function of ell, k, eta
    def gen_coeff(l1,l2,l3):
        def gen_neg(l1,l2,l3):
            L=(l1+l2+l3)/2+l1+l2
            ev=((l1+l2+l3) % 2) ==0
            exp=L[ev]
            neg=np.zeros_like(L)
            neg[ev]=(-1)**exp
            return neg
        return gen_neg(l1,l2,l3)*(2*l3+1)*(2*l1+1)*(2*l2+1)
    l_1=l1[:,np.newaxis,np.newaxis]
    l_2=l2[np.newaxis,:,np.newaxis]
    l_3=l3[np.newaxis,np.newaxis,:]
    coeff=gen_coeff(l_1,l_2,l_3)*wig3j**2
    T_integrand=np.zeros([l3.size,interparr.shape[0],interparr.shape[1]])
    print('starting l1 l2 sum')
    for i in np.arange(l2.size):
        jn=gen_bess_fast(l2[i],interparr)
        for j in np.arange(l1.size):
            thiscoeff=coeff[j,i,:]
            nonz=(thiscoeff != 0)
            if np.any(nonz):
                T_integrand[nonz,:,:]+=(jn*T_1_arr[j,:,:])[np.newaxis,:,:]*(thiscoeff[nonz,np.newaxis,np.newaxis])
        if not (i % 100): print(('%s of %s' % (i, l2.size)))
    return T_integrand/(2*l3+1)[:,np.newaxis,np.newaxis]


def gen_jn_jnp(tau,kint,ell): #pretabulate bessel functions
    jn=np.zeros([ell.size,tau.size,kint.size])
    jnp=np.copy(jn)
    interparr=(eta_0-tau)[:,np.newaxis]*kint[np.newaxis,:]
    for i in range(ell.size):
        jn[i,:,:],jnp[i,:,:]=gen_bess_fast(ell[i],interparr,derivative=True)
        if not (i % 10): print(('%s of %s' % (i,ell.size)))
    return jn,jnp

#function generating mu nu and gamma
def gen_1d_k_parts(tau,kint,ell,del_e,del_ell,J_ell,jn_arr,jnp_arr):
    lnk=np.log(kint)
    Parrkcube=As*(kint/kpivot)**(ns-1.)*(2*np.pi**2)
    interparr=(eta_0-tau)[:,np.newaxis]*kint[np.newaxis,:]
    del_e_l_P=del_ell[:,np.newaxis,:]*del_e*Parrkcube
    gam=np.zeros([ell.size,tau.size])
    mu=np.copy(gam)
    nu=np.copy(gam)
    for i in range(ell.size):
        jn=jn_arr[i]
        jnp=jnp_arr[i]
        quad=del_ell[i,np.newaxis,:]*Parrkcube*J_ell[i,:,:]
        gam[i,:]=trap_int(quad,lnk,axis=1)
        quad=jn/interparr*del_e_l_P[i,:,:]
        quad[interparr==0.]=0.#del_e_l_P[i,:,:][interparr==0.]
        mu[i,:]=trap_int(quad,lnk,axis=1)
        quad=jnp*del_e_l_P[i,:,:]
        nu[i,:]=trap_int(quad,lnk,axis=1)
        if not (i % 10): print(('%s of %s' % (i,ell.size)))
    return mu/(2*np.pi**2),nu/(2*np.pi**2),gam/(2*np.pi**2) #factors of 4pi from angular integral being restored


def gen_A_B(tau,ell,mu,nu,gam):
    quad_A=-g_int(tau)*nu[:,np.newaxis,:]*nu[np.newaxis,:,:] #nu^2 results in a negative (Del_e has a factor -i)
    quad_B=-g_int(tau)*mu[:,np.newaxis,:]*mu[np.newaxis,:,:] #mu^2 results in a negative 
    A=np.zeros([ell.size,ell.size,ell.size])
    B=np.copy(A)
    for i in range(ell.size):
        A[:,:,i]=trap_int(quad_A*gam[i,:],tau,axis=2)
        B[:,:,i]=trap_int(quad_B*gam[i,:],tau,axis=2)
        if not (i % 10): print(('%s of %s' % (i,ell.size)))
    return A*2*(4*np.pi)**3,B*2*(4*np.pi)**3



#get_ipython().run_cell_magic('cython', '', '\nfrom cython cimport cdivision,boundscheck\n\ncdef extern from "math.h":\n    int abs(int x)\n\n@cdivision\n@boundscheck(False) #Gen \\mathcal{Q}^2: g_ell_sq is g_l1,l2,l3 ^2\ndef gen_Qsq(double[:,:,:] g_ell_sq, int[:] ell, int[:] ellfull, double[:,:,:,:] retval):\n    cdef int ellmax=ell.shape[0],ellfullmax=ellfull.shape[0]\n    cdef int ellmast, ell_1,ell_2,ell_3,ell_4\n    while (ell_4 < ellmax):\n        while (ell_3 < ellmax):\n            while (ell_2 < ellmax):\n                while (ell_1 < ellmax):\n                    while (ellmast < ellfullmax):\n                        if (g_ell_sq[ell_1,ell_2,ellmast]!=0.)*(g_ell_sq[ell_3,ell_4,ellmast]!=0.):\n                            retval[ell_1,ell_2,ell_3,ell_4]+=g_ell_sq[ell_1,ell_2,ellmast]*g_ell_sq[ell_3,ell_4,ellmast]/(2.*ellfull[ellmast]+1)\n                        ellmast+=1\n                    ellmast=0\n                    ell_1+=1\n                ell_1=0\n                ell_2+=1\n            ell_2=0\n            ell_3+=1\n        ell_3=0\n        ell_4+=1\n    return retval\n\n@cdivision\n@boundscheck(False) #Gen \\mathcal{Q}\\tilde{\\mathcal{Q}}: g_til_ell is \\tilde{g}_l1,l2,l3\ndef gen_QQt(double[:,:,:] g_ell,double[:,:,:] g_til_ell, int[:] ell, int[:] ellfull, double[:,:,:,:] retval):\n    cdef int ellmax=ell.shape[0],ellfullmax=ellfull.shape[0]\n    cdef int ellmast, ell_1,ell_2,ell_3,ell_4\n    while (ell_4 < ellmax):\n        while (ell_3 < ellmax):\n            while (ell_2 < ellmax):\n                while (ell_1 < ellmax):\n                    while (ellmast < ellfullmax):\n                        if (g_ell[ell_1,ell_2,ellmast]!=0.)*(g_ell[ell_3,ell_4,ellmast]!=0.)*(g_til_ell[ell_1,ell_2,ellmast]!=0.):\n                            retval[ell_1,ell_2,ell_3,ell_4]+=g_ell[ell_1,ell_2,ellmast]*g_til_ell[ell_1,ell_2,ellmast]*g_ell[ell_3,ell_4,ellmast]*g_ell[ell_3,ell_4,ellmast]/(2.*ellfull[ellmast]+1)\n                        ellmast+=1\n                    ellmast=0\n                    ell_1+=1\n                ell_1=0\n                ell_2+=1\n            ell_2=0\n            ell_3+=1\n        ell_3=0\n        ell_4+=1\n    return retval\n\n@cdivision\n@boundscheck(False) #Gen \\tilde{\\mathcal{Q}}^2:\ndef gen_Qtsq(double[:,:,:] g_ell_sq,double[:,:,:] g_til_ell_sq, int[:] ell, int[:] ellfull, double[:,:,:,:] retval):\n    cdef int ellmax=ell.shape[0],ellfullmax=ellfull.shape[0]\n    cdef int ellmast, ell_1,ell_2,ell_3,ell_4\n    while (ell_4 < ellmax):\n        while (ell_3 < ellmax):\n            while (ell_2 < ellmax):\n                while (ell_1 < ellmax):\n                    while (ellmast < ellfullmax):\n                        if (g_til_ell_sq[ell_1,ell_2,ellmast]!=0.)*(g_ell_sq[ell_3,ell_4,ellmast]!=0.):\n                            retval[ell_1,ell_2,ell_3,ell_4]+=g_til_ell_sq[ell_1,ell_2,ellmast]*g_ell_sq[ell_3,ell_4,ellmast]/(2.*ellfull[ellmast]+1)\n                        ellmast+=1\n                    ellmast=0\n                    ell_1+=1\n                ell_1=0\n                ell_2+=1\n            ell_2=0\n            ell_3+=1\n        ell_3=0\n        ell_4+=1\n    return retval\n\n\n@cdivision\n@boundscheck(False) #Gen \\tilde{\\mathcal{Q}}\\tilde{\\mathcal{Q}}^T\ndef gen_Qt_trans(double[:,:,:] g_ell,double[:,:,:] g_til_ell, int[:] ell, int[:] ellfull, double[:,:,:,:] retval):\n    cdef int ellmax=ell.shape[0],ellfullmax=ellfull.shape[0]\n    cdef int ellmast, ell_1,ell_2,ell_3,ell_4\n    while (ell_4 < ellmax):\n        while (ell_3 < ellmax):\n            while (ell_2 < ellmax):\n                while (ell_1 < ellmax):\n                    while (ellmast < ellfullmax):\n                        if (g_til_ell[ell_1,ell_2,ellmast]!=0.)*(g_ell[ell_3,ell_4,ellmast]!=0.):\n                            retval[ell_1,ell_2,ell_3,ell_4]+=g_ell[ell_1,ell_2,ellmast]*g_til_ell[ell_1,ell_2,ellmast]*g_ell[ell_3,ell_4,ellmast]*g_til_ell[ell_3,ell_4,ellmast]/(2.*ellfull[ellmast]+1)\n                        ellmast+=1\n                    ellmast=0\n                    ell_1+=1\n                ell_1=0\n                ell_2+=1\n            ell_2=0\n            ell_3+=1\n        ell_3=0\n        ell_4+=1\n    return retval\n\n@cdivision\n@boundscheck(False) #Gen \\tilde{\\mathcal{Q}}\\tilde{\\mathcal{Q}}^S: g_ell_s is g_{l,l1,l2}^{s,-s,0}, g_ell_ms is g_{l,l1,l2}^{-s,s,0}\ndef gen_Qt_scat(double[:,:,:] g_ell,double[:,:,:] g_til_ell,double[:,:,:] g_ell_s,double[:,:,:] g_ell_ms, int[:] ell, int[:] ellfull, double[:,:,:,:] retval):\n    cdef int ellmax=ell.shape[0],ellfullmax=ellfull.shape[0]\n    cdef int ellmast, ell_1,ell_2,ell_3,ell_4\n    while (ell_4 < ellmax):\n        while (ell_3 < ellmax):\n            while (ell_2 < ellmax):\n                while (ell_1 < ellmax):\n                    while (ellmast < ellfullmax):\n                        if (g_til_ell[ell_1,ell_2,ellmast]!=0.)*(g_ell[ell_3,ell_4,ellmast]!=0.):\n                            retval[ell_1,ell_2,ell_3,ell_4]+=g_til_ell[ell_1,ell_2,ellmast]*(g_ell_s[ellmast,ell_1,ell_2]*g_ell_ms[ellmast,ell_3,ell_4]+g_ell_ms[ellmast,ell_1,ell_2]*g_ell_s[ellmast,ell_3,ell_4])*g_ell[ell_3,ell_4,ellmast]/(2.*ellfull[ellmast]+1)\n                        ellmast+=1\n                    ellmast=0\n                    ell_1+=1\n                ell_1=0\n                ell_2+=1\n            ell_2=0\n            ell_3+=1\n        ell_3=0\n        ell_4+=1\n    return retval')



def gen_alph(Del_ell,ell,k,r,interparr,gen_beta=True):
    kcube=k**3
    dlnk=np.log(k[1:]/k[:-1])
    alph=np.zeros([ell.size,r.size])
    if gen_beta: 
        Parr=As*(k/kpivot)**(ns-1)*(2*np.pi**2)/kcube
        beta=np.zeros([ell.size,r.size])
    for i in range(ell.size):
        jn=gen_bess_fast(ell[i],interparr)
        quad=Del_ell[i,:]*jn*kcube
        if gen_beta: 
            quadbeta=quad*Parr
            beta[i,:]=((quadbeta[:,1:]+quadbeta[:,:-1])*dlnk).sum(axis=1)/2.
        alph[i,:]=((quad[:,1:]+quad[:,:-1])*dlnk).sum(axis=1)/2.
        if not (i%10): print(('%s of %s' % (i, ell.size)))
    if gen_beta:
        return alph*10/3./np.pi, beta*6/5./np.pi
    return alph*10/3./np.pi



#get_ipython().run_cell_magic('cython', '', '\nfrom cython cimport cdivision,boundscheck\n\n@cdivision\n@boundscheck(False)\ncdef double r_int(double dr1, double dr2, double[:] r, double[:] alph, double[:] beta1,double[:] beta2, double[:] beta3, int[:] indswitch):\n    cdef int rmax=r.shape[0],rcount=1\n    cdef double rint, rint2\n    rint+=alph[0]*beta1[0]*beta2[0]*beta3[0]*r[0]*r[0]/2.\n    rint+=alph[rmax-1]*beta1[rmax-1]*beta2[rmax-1]*beta3[rmax-1]*r[rmax-1]*r[rmax-1]/2.\n    rint+=alph[indswitch[0]]*beta1[indswitch[0]]*beta2[indswitch[0]]*beta3[indswitch[0]]*r[indswitch[0]]*r[indswitch[0]]/2.\n    rint2+=alph[indswitch[0]]*beta1[indswitch[0]]*beta2[indswitch[0]]*beta3[indswitch[0]]*r[indswitch[0]]*r[indswitch[0]]/2.\n    rint2+=alph[indswitch[1]]*beta1[indswitch[1]]*beta2[indswitch[1]]*beta3[indswitch[1]]*r[indswitch[1]]*r[indswitch[1]]/2.\n    rint+=alph[indswitch[1]]*beta1[indswitch[1]]*beta2[indswitch[1]]*beta3[indswitch[1]]*r[indswitch[1]]*r[indswitch[1]]/2.\n    while (rcount < rmax-2):\n        if ((rcount < indswitch[0]) or (rcount > indswitch[1])):\n            rint+=alph[rcount]*beta1[rcount]*beta2[rcount]*beta3[rcount]*r[rcount]*r[rcount]\n        elif ((rcount > indswitch[0]) or (rcount < indswitch[1])):\n            rint2+=alph[rcount]*beta1[rcount]*beta2[rcount]*beta3[rcount]*r[rcount]*r[rcount]\n        rcount+=1\n    return rint*dr1+rint2*dr2\n    \n\n@cdivision\n@boundscheck(False)\ndef gen_C_ellx4_slow(double[:,:] alph, double[:,:] beta, int[:] ell, double[:] r, double dr1, double dr2, int[:] indswitch, double[:,:,:,:] retval):\n    cdef int ellmax=ell.shape[0],rmax=r.shape[0]\n    cdef int ell_1,ell_2,ell_3,ell_4\n    #cdef double[:] r1=r[:indswitch[0]], double[:,:] alph1=alph[:,:indswitch[0]], double[:,:] beta1=beta[:,:indswitch[0]]\n    #cdef double[:] r2=r[indswitch[1]:indswitch[2]], double[:,:] alph2=alph[:,indswitch[0]:indswitch[1]], double[:,:] beta2=beta[:,indswitch[0]:indswitch[1]]\n    #cdef double[:] r3=r[indswitch[1]:], double[:,:] alph3=alph[:,indswitch[1]:], double[:,:] beta3=beta[:,indswitch[1]:]\n    while (ell_4 < ellmax):\n        while (ell_3 < ellmax):\n            while (ell_2 < ellmax):\n                while (ell_1 < ellmax):\n                    retval[ell_1,ell_2,ell_3,ell_4]+=r_int(dr1,dr2,r,alph[ell_4,:],beta[ell_1,:],beta[ell_2,:],beta[ell_3,:],indswitch)\n                    ell_1+=1\n                ell_1=0\n                ell_2+=1\n            ell_2=0\n            ell_3+=1\n        ell_3=0\n        ell_4+=1\n    return retval\n         ')


def comp_N_ell(varlist, FWHMlist,ell):
    expterm=2*np.log(((varlist*FWHMlist/T0)))+(ell*(ell+1))[:,np.newaxis]*FWHMlist**2/8/np.log(2)
    N_ell=np.exp(-expterm)
    return 1/(N_ell.sum(axis=1))


comm = MPI.COMM_WORLD
rank=comm.Get_rank()


# # Generate $C[\Theta]_{\ell_2}$

# In[2]:
if rank==0:
    with open('PBH_prog.txt','w') as f:
        print(('starting code'),file=f)

l_max_g=100 # max ell taken for the collision term
L=np.arange(l_max_g)


if (rank!=1) or (rank==1):
# In[3]:



    common_settings = {# wich output? ClTT, transfer functions delta_i and theta_i
                           'output':'tCl,pM,vTk,mTk',
                           # LambdaCDM parameters
                           'h':h,
                            'input_verbose':4,
                            'background_verbose':4,
                            'T_cmb':T0,
                           'Omega_b':omegab,
                           'Omega_cdm':omegaM-omegab,
                           'modes':'s',
                           #'tensor method':'photons',
                           #'k_output_values':kstr,
                           'N_ur':3.046,
                           'A_s':As,
                           'n_s':ns,
                            #'sigma8':0.8,
                           #'k_output_values':'1e-3,1e-2,1e-1',
                           'tau_reio':tau_reio,
                           #'primordial_P_k_max_1/Mpc':10,
                            #'z_max_pk':2000.,
                            #'k_per_decade_for_pk':100,
                           #'k_per_decade_for_bao':100,
                           # Take fixed value for primordial Helium (instead of automatic BBN adjustment)
                           'YHe':YHe,
                           'l_max_g':l_max_g+30,
                           'l_max_pol_g':l_max_g+10,
                           'l_max_scalars':8000,
                           'k_pivot':kpivot,
                           'radiation_streaming_trigger_tau_over_tau_k':1e6,
                           'radiation_streaming_trigger_tau_c_over_tau':1e6,
                           #'ur_fluid_trigger_tau_over_tau_k':10,
                           # other output and precision parameters
                           #'P_k_max_1/Mpc':10.0,
                           'recombination':'HyRec',
                           'gauge':'newtonian',
                           'lensing':'no'}

    M = Class()
    M.struct_cleanup()  # clean output
    M.empty()
    M.set(common_settings)
    M.compute()
    s,k,tau = M.get_sources()

    with open('C_ell_prog.txt','w') as f:
        print(('computed class quantities'),file=f)

    print(('computed class quantities'))

    # # Interpolate thermodynamics from CLASS

    # In[4]:


    therm = M.get_thermodynamics()
    g_class=therm['g [Mpc^-1]']
    eta_class=therm['conf. time [Mpc]']
    g_int=interp1d(eta_class,g_class,fill_value=0.,bounds_error=False)#,kind=interpkind)
    eta_int=interp1d(therm['z'],eta_class)
    xe_class=interp1d(therm['z'],therm['x_e'])
    eta_0=M.get_background()['conf. time [Mpc]'][-1]
    z_int=interp1d(eta_class,therm['z'],bounds_error=False,fill_value=-1)
    kappadot_int=interp1d(eta_class,therm["kappa' [Mpc^-1]"])


# # Load in the $\ell$ and $k$ resolution that are sufficient for the homogeneous $C^{(TT)}$.

# In[5]:


# trsf=np.loadtxt('/Users/Acolyte/NYU/Research/CLASS/class_extract_l/output/T_transf/test.trsf')
# l_trsf=trsf[1:,0].astype('int')
# k_trsf=trsf[0,1:]
# trsf=trsf[1:,1:]
l_trsf=np.loadtxt('./data/l_trsf.dat').astype(int)
k_trsf=np.loadtxt('./data/k_trsf.dat')
kint=k_trsf
# In[6]:

tmp =((l_trsf[1:]+l_trsf[:-1])/2.).astype('int')
l_trsf=np.unique(np.sort(np.append(tmp,l_trsf))).astype(int)

keps=6e-3/3.*2
knot=10e-5/3.*2
kmax=3/2.*5000/eta_0
klogcut=knot/keps
kint=10**np.arange(-5,np.log10(klogcut),keps)
kint=np.append(kint,np.arange(klogcut,kmax,knot))
k_trsf=kint



# # Compute $\Delta_\ell$ via line of sight

# In[10]:





# In[11]:
if rank==0:
    with open('C_ell_prog.txt','w') as f:
        print(('compute del_ell'),file=f)

    print('compute del_ell')

    S0=gen_S0(s,k,tau) #generate the source functions.
    kint=k_trsf
    x=10**(np.linspace(-3,5,12000)) #expedite this calculation by defining x = k\chi
    
    tauint=eta_0-x[:,np.newaxis]/kint #compute tau from x
    S0_int=S0_interp_2d(S0,k,tau,kint,tauint) #with new k and tau array, interpolate the source functions
    interparr=(eta_0-tau)[:,np.newaxis]*kint #this is the bessel function's argument
    # In[12]:
    
    
    #compute each multipole of temperature anisotropy transfer function
    T_l_0_2d=np.zeros([l_trsf.size,kint.size])
    for i in range(l_trsf.size):
        T_l_0_2d[i,:]=single_l(tauint,x,S0_int,l_trsf[i])
        if not (i % 20): print(('%s of %s' % (i,l_trsf.size)))


taulog=1e-3/3.*2
taulin=50/3.*2

tauhi=np.exp(np.arange(np.log(eta_int(1400)),np.log(eta_int(900)),taulog))
tauhi=np.append(tauhi,np.exp(np.arange(np.log(eta_int(900)),np.log(eta_int(10)),20*taulog)))
tauhi=np.append(tauhi,np.arange(eta_int(10),eta_int(0),taulin))

# tmp =((tau[1:]+tau[:-1])/2.)
# tauhi=np.unique(np.sort(np.append(tmp,tau)))
# # In[13]:


# dlnk=np.log(kint[1:]/kint[:-1])
# Parrkcube=As*(kint/kpivot)**(ns-1.)*(2*np.pi**2)

# quad=T_l_0_2d**2*Parrkcube
# #quad=T_l_0*T_l_0_nocoeff
# C_l_comp_2d=((quad[:,1:]+quad[:,:-1])*dlnk).sum(axis=1)/np.pi


# # # Compute $\Delta_e$

# # In[14]:


# #Set the PBH parameters
# pbhmass=100 #in solar masses
# PI=False #Photoionization or collisionally ionized?
# fpbh='1' #fraction of f_pbh, these are hard coded for the time being
# if PI: PI_str='_PI'
# else: PI_str=''
# appstr='_mbh%s_f%s%s' % (pbhmass, fpbh,PI_str) #naming scheme for data


# # bin parameters that are hard-coded from radiation transport sims

# # In[15]:


# #Define bins used for simulations
# afinal=1/51.
# zfinal=50
# astartb=1/1501.

# astep=0.01
# abins=(np.arange(np.log(astartb),np.log(afinal),astep))
# tp=abins+astep/4.
# tm=abins-astep/4.
# abins=np.empty((tp.size+tm.size),dtype=tp.dtype)
# abins[0::2]=np.exp(tm)
# abins[1::2]=np.exp(tp)

# nrbins=140
# logr=np.linspace(np.log(1),np.log(1000),nrbins)
# rbins=np.insert(np.exp(logr),0,0)
# rbins=np.append(rbins,100000) #excise this nonsense

# amean=(abins[1:]+abins[:-1])/2.


# astartb=1/1501.
# afininj=1/101.
# aliststep=0.025
# alistiter=np.exp(np.arange(np.log(astartb),np.log(afininj),aliststep))
# alistiterdub=np.exp(np.arange(np.log(astartb)-aliststep/2.,np.log(afininj)-aliststep/2.,aliststep/2.))
# alistiter=alistiterdub



# # Functions to generate $G^{\rm inj}_{x_e}$

# # In[16]:





# #Load in data into arrays (r,adep,ainj)



# # Generate Fourier transformed $G_{\rm dep}^{\rm inj}$ 

# # In[17]:




# #karr=np.exp(np.linspace(np.log(0.001),np.log(1.),200))
# fol='./data/fixed_03_2021/'
# ardep,nphotbin,adist,Etot=load_tab(alistiter,abins, rbins,fol,PI=PI,dlnanam=0.0025,suffix='fixed.pkl') #tab_frac_bin.pkl
# k_G=np.exp(np.linspace(np.log(0.0001),np.log(100.),100))
# G_homo,G_k=gen_greens(alistiter,abins,ardep,nphotbin,adist,Etot,k_G)#(adep,ainj) (k,adep,ainj)


# # Generate $G_{x_e}^{\rm dep}$

# # In[18]:



# numz=150 #resolution of G_xe^dep (I end up interpolating this anyway)
# Gtfil='./data/Gxe_analytic_adep_same.dat'
# G_xe, z=load_G_xe(Gtfil,numz=numz)
# alog_inj=np.log(alistiter)
# alog_dep=(np.log(abins)[1:]+np.log(abins)[:-1])/2.
# zinj=1/np.exp(alog_inj)-1
# zdep=1/np.exp(alog_dep)-1



# # In[19]:




# # In[20]:

# with open('PBH_prog.txt','w') as f:
#     print(('computing del_e'),file=f)


# print('computing del_e')

# W=gen_W(alog_inj,alog_dep,z,G_homo,G_xe,G_k) #(z, a_inj,k) G_xe^inj


# # Load in transfer files and generate the bias

# # In[21]:


# transf_file='./data/tcb_alistiterdub.pkl'
# transf=load_tcb(transf_file)
# vrmsval=vrms(transf['TF'],transf['k'])
# zind=find_nearest(transf['z_inj'],800) #choosing random redshift <1000 for spatial signature
# testk=transf['k']
# testk[0]-=4.1594e-14 #rounding error causes interpolation outside of range, this fixes it
# transf_int=interp1d(testk,transf['TF'][zind,:]/vrmsval[zind]) #this is the spatial part of the transfer function
# b=gen_b(pbhmass,alog_inj,vrmsval,f=fpbh,PI=PI)


# # Compute $G_e$ as defined in the paper. 
# # 
# # 
# # $\int (v_{\rm bc})^2 G_e d\chi\\
# # G_e(\eta, \eta', k')\equiv G_{x_e}^{\rm inj}(\eta, \eta', k|\Psi) \frac{\Gamma(\eta')}{x_e^0(\eta)},\\
# # \Gamma(\eta)\equiv \frac{\overline{\epsilon}_{\rm inj}aHb}{\langle{v^2_{\rm bc}}\rangle }.$

# # In[22]:


# quad=W*(b*vrmsval**2)[np.newaxis,:,np.newaxis] #integral over lna, hence the lack of the hubblerate and factor of a
# #b already has eps_inj and <v^2_bc>, we multiply by the latter (the amplitude of vbc transfer) to compute G_e integrated over eta'
# dlna=alog_inj[1:]-alog_inj[:-1]
# G_e=-((quad[:,1:,:]+quad[:,:-1,:])*dlna[np.newaxis,:,np.newaxis]).sum(axis=1)/2./xe_class(z)[:,np.newaxis] #the minus sign is to make it positive, for when I squareroot At the end of the day, Del_e should have a factor of (-i) to make it negative when squaring.


# # Interpolate and generate the $\Delta_e$ by approximating $G_e(|k_1+k_2|)\approx G_e(\sqrt{2}k_1)G_e(\sqrt{2}k_2)$

# # In[23]:




# Del_e=interp_G_e_new(G_e,z,k_G,tau,kint) #Because of the minus sign, this is technically multiplied by an (i)


# # Gen $\mathcal{J_\ell}$

# # $\mathcal{J}_\ell(k,\chi)\equiv\frac{4\pi}{(2\ell+1)} \sum_{\ell_1\ell_2}\!(-1)^{L}(g_{\ell_1\ell_2\ell})^2j_{\ell_1}( k,\chi)C^{(0)}_{\ell_2}(k)\\
# L\equiv 3(\ell_1+\ell_2)/2+\ell/2$

# In[24]:



# In[25]:

with open('PBH_prog.txt','w') as f:
    print(('computing J_ell'),file=f)


print('computing J_ell')

l3=l_trsf
l3p=gen_l_arr(l3,l_max_g)


# In[26]:

if rank==1:
    kcut=(k.min()<kint)*(k.max()>kint)
    Cthet=np.swapaxes(gen_Cthet_arr_hi(s,tau,tauhi,k,kint[kcut],L),1,2) #l,k,tau swapped to l,tau,k
    
    
    # In[27]:
    
    
    wig3j=gen_wig3j(L,l3p,l3)
    
    
    # In[28]:
    
    
    interparr=kint[np.newaxis,kcut]*(eta_0-tauhi)[:,np.newaxis]
    l_mark=1
    jl=spherical_jn(1,interparr)
    jl_n=spherical_jn(0,interparr)
    J_ell=gen_J_ell_py(Cthet,interparr,L,l3p,l3,wig3j=wig3j)
    
    
    # In[39]:
    
    
    with open('./J_ell_%s_hi_times2.pkl' % (l_max_g),"wb") as f:
        pickle.dump(J_ell,f)
        pickle.dump(l_trsf,f)
        pickle.dump(tauhi,f)
        pickle.dump(kint[kcut],f)


# In[21]:


# with open('./J_ell.pkl',"rb") as f:
#     J_ell=pickle.load(f)
#     l_trsf=pickle.load(f)
#     tau=pickle.load(f)
#     #k_trsf=pickle.load(f)


# # Generate $\mu$,  $\nu$, $\gamma$

# # $\mu_\ell(\chi) \equiv \int Dk~ P(k)~ \Delta_{\ell}(k) \frac{j_\ell(\chi k)}{\chi k} \Delta_e(\chi, k),  \\
# \nu_\ell(\chi) \equiv \int Dk~ P(k)~ \Delta_{\ell}(k) j_\ell'(\chi k) \Delta_e(\chi, k), \\
# \gamma_\ell(\chi) \equiv \int Dk~ P(k)~ \Delta_{\ell}(k) \mathcal{J}_\ell(\chi, k).$

# In[29]:


        


# In[30]:


# kcut=(k.min()<kint)*(k.max()>kint)
# interparr=kint[np.newaxis,kcut]*(eta_0-tau)[:,np.newaxis]
# l_mark=1
# jl=spherical_jn(1,interparr)
# jl_n=spherical_jn(0,interparr)
# jn_arr,jnp_arr=gen_jn_jnp(tau,kint[kcut],l_trsf)


# # In[31]:

# with open('PBH_prog.txt','w') as f:
#     print(('computing cal_A and cal_B'),file=f)


# print('computing cal_A and cal_B')

# mu,nu,gam=gen_1d_k_parts(tau,kint[kcut],l_trsf,Del_e[:,kcut],T_l_0_2d[:,kcut],J_ell,jn_arr,jnp_arr) #ell, tau


# # In[32]:


# cal_A,cal_B=gen_A_B(tau,l_trsf,mu,nu,gam)


# # In[33]:


# del mu, nu, gam, J_ell


# # Compute Geometric Quantities 

# # $(\mathcal{Q}^2)_{\ell_1 \ell_2 \ell_3 \ell_4} = \sum_{\ell} \frac1{2 \ell +1} (g_{\ell_1 \ell_2 \ell})^2 (g_{\ell \ell_3 \ell_4})^2, \\
# (\mathcal{Q \widetilde{Q}})_{\ell_1 \ell_2, \ell_3 \ell_4} = - \frac12 \sqrt{\ell_1 (\ell_1 +1) \ell_2 (\ell_2 +1)} \sum_{\ell}\frac1{2 \ell +1} g_{\ell_1 \ell_2 \ell}~\widetilde{g}_{\ell_1 \ell_2, \ell} (g_{\ell_3 \ell_4 \ell})^2, \\
# (\mathcal{\widetilde{Q}}^2)_{\ell_1 \ell_2, \ell_3 \ell_4} = \frac14 \ell_1 (\ell_1 +1) \ell_2 (\ell_2 +1) \sum_{\ell} \frac1{2 \ell +1} (\widetilde{g}_{\ell_1 \ell_2, \ell})^2 (g_{\ell \ell_3 \ell_4})^2, \\
# (\mathcal{\widetilde{Q} \widetilde{Q}}^{\rm T})_{\ell_1 \ell_2, \ell_3 \ell_4} =  \frac14 \sqrt{\ell_1 (\ell_1 +1) \ell_2 (\ell_2 +1)} \sqrt{\ell_3 (\ell_3 +1) \ell_4 (\ell_4 +1)} \sum_{\ell} \frac1{2 \ell +1} (g_{\ell_1 \ell_2 \ell} ~\widetilde{g}_{\ell_1 \ell_2, \ell}) (g_{\ell_3 \ell_4 \ell} ~\widetilde{g}_{\ell_3 \ell_4, \ell}), \\
#  (\mathcal{\widetilde{Q} \widetilde{Q}}^{\rm S})_{\ell_1, \ell_2 \ell_3, \ell_4} = - \frac14 \ell_1 (\ell_1 +1) \sqrt{\ell_2(\ell_2 +1) \ell_3 (\ell_3 + 1)} \sum_{s = \pm 1} \sum_{\ell} \frac1{2 \ell +1}\widetilde{g}_{\ell_1 \ell_2, \ell} g_{\ell, \ell_1, \ell_2}^{s, -s, 0} ~g_{\ell, \ell_3 ,\ell_4}^{-s, s, 0}~g_{\ell \ell_3 \ell_4}.$
# 

# In[34]:


#%load_ext wurlitzer


# In[35]:




# In[36]:

if rank==-1:
    with open('PBH_prog.txt','w') as f:
        print(('computing g_ells for Q geo'),file=f)


print('computing g_ells for Q geometric')

ell=l_trsf
ellfull=np.arange(2*ell.max()+1)


# In[37]:


#gen g_{l1 l2 l3}
if rank==-1:
    g_ell=gen_wig3j(ell,ell,ellfull,spin=0)*np.sqrt((2*ell[:,np.newaxis,np.newaxis]+1)*(2*ell[np.newaxis,:,np.newaxis]+1)*(2*ellfull[np.newaxis,np.newaxis,:]+1)/(4*np.pi))
    #gen g_{l1 l2 l}^{s -s 0}
    #g_ell_sp=gen_wig3j(ell,ell,ellfull,spin=1)*np.sqrt((2*ell[:,np.newaxis,np.newaxis]+1)*(2*ell[np.newaxis,:,np.newaxis]+1)*(2*ellfull[np.newaxis,np.newaxis,:]+1)/(4*np.pi))
    #gen \tilde{g}_{l1 l2 l3}
    #l_mast=ell[:,np.newaxis,np.newaxis]+ell[np.newaxis,:,np.newaxis]+ellfull[np.newaxis,np.newaxis,:]
    #g_til_ell=np.where((l_mast % 2).astype(bool),0.,2*g_ell_sp)
    with open('./g_til_ell.pkl',"rb") as f:
        g_til_ell=pickle.load(f, encoding="latin1")
        
    #gen g_{l l1 l2}^{s -s 0} and g_{l l1 l2}^{-s s 0}
    #g_ell_s=gen_wig3j(ellfull,ell,ell,spin=1)*np.sqrt((2*ellfull[:,np.newaxis,np.newaxis]+1)*(2*ell[np.newaxis,:,np.newaxis]+1)*(2*ell[np.newaxis,np.newaxis,:]+1)/(4*np.pi))
    with open('./g_ell_s.pkl',"rb") as f:
        g_ell_s=pickle.load(f, encoding="latin1")
        
    l_mast=ellfull[:,np.newaxis,np.newaxis]+ell[np.newaxis,:,np.newaxis]+ell[np.newaxis,np.newaxis,:]
    g_ell_ms=g_ell_s*(-1)**(l_mast)


# In[41]:
if rank ==-1:
    with open('PBH_prog.txt','w') as f:
        print(('computing Qsq'),file=f)
    print('computing Qsq')
    retval=np.zeros([ell.size,ell.size,ell.size,ell.size])
    Qsq=np.asarray(gen_Qsq(g_ell**2, ell.astype(np.int32), ellfull.astype(np.int32), retval))
    with open('./Qsq.pkl',"wb") as f:
        pickle.dump(Qsq,f)
        pickle.dump(ell,f)


# In[42]:



if rank ==-1:
    print('computing QQt')

    retval=np.zeros([ell.size,ell.size,ell.size,ell.size])
    QQtcoeff=-np.sqrt((ell*(ell+1))[:,np.newaxis,np.newaxis,np.newaxis]*(ell*(ell+1))[np.newaxis,:,np.newaxis,np.newaxis])/2.
    QQt=np.asarray(gen_QQt(g_ell, g_til_ell, ell.astype(np.int32), ellfull.astype(np.int32), retval))*QQtcoeff
    with open('./QQt.pkl',"wb") as f:
        pickle.dump(QQt,f)
        pickle.dump(ell,f)




# In[43]:


if rank ==-1:
    print('computing Qtsq')


    retval=np.zeros([ell.size,ell.size,ell.size,ell.size])
    Qtsqcoeff=((ell*(ell+1))[:,np.newaxis,np.newaxis,np.newaxis]*(ell*(ell+1))[np.newaxis,:,np.newaxis,np.newaxis])/4.
    Qtsq=np.asarray(gen_Qtsq(g_ell**2, g_til_ell**2, ell.astype(np.int32), ellfull.astype(np.int32), retval))*Qtsqcoeff
    with open('./Qtsq.pkl',"wb") as f:
        pickle.dump(Qtsq,f)
        pickle.dump(ell,f)



# In[44]:

if rank ==-2:
    print('computing Qt_trans')


    retval=np.zeros([ell.size,ell.size,ell.size,ell.size])
    Qt_transcoeff=np.sqrt((ell*(ell+1))[:,np.newaxis,np.newaxis,np.newaxis]*(ell*(ell+1))[np.newaxis,:,np.newaxis,np.newaxis])*np.sqrt((ell*(ell+1))[np.newaxis,np.newaxis,:,np.newaxis]*(ell*(ell+1))[np.newaxis,np.newaxis,np.newaxis,:])/4.
    Qt_trans=np.asarray(gen_Qt_trans(g_ell, g_til_ell, ell.astype(np.int32), ellfull.astype(np.int32), retval))*Qt_transcoeff
    with open('./Qt_trans.pkl',"wb") as f:
        pickle.dump(Qt_trans,f)
        pickle.dump(ell,f)


# In[45]:

if rank ==-3:
    with open('PBH_prog.txt','w') as f:
        print(('computing Qt_scat'),file=f)
    print('computing Qt_scat')
    retval=np.zeros([ell.size,ell.size,ell.size,ell.size])
    Qt_scatcoeff=-(ell*(ell+1))[:,np.newaxis,np.newaxis,np.newaxis]*np.sqrt((ell*(ell+1))[np.newaxis,:,np.newaxis,np.newaxis]*(ell*(ell+1))[np.newaxis,np.newaxis,:,np.newaxis])/4.
    Qt_scat=np.asarray(gen_Qt_scat(g_ell, g_til_ell, g_ell_s, g_ell_ms,ell.astype(np.int32), ellfull.astype(np.int32), retval))*Qt_scatcoeff
    with open('./Qt_scat.pkl',"wb") as f:
        pickle.dump(Qt_scat,f)
        pickle.dump(ell,f)



# In[46]:

if rank==-1:
    del g_ell, g_til_ell, g_ell_s, g_ell_ms, l_mast


# In[52]:


# In[54]:


# with open('./Q_var.pkl',"wb") as f:
#    pickle.dump(Qtsq,f)
#    pickle.dump(Qt_trans,f)
#    pickle.dump(Qt_scat,f)
#    pickle.dump(ell,f)


# In[26]:


# with open('./Q_bias.pkl',"rb") as f:
#     Qsq=pickle.load(f)
#     QQt=pickle.load(f)
#     ell=pickle.load(f)


# In[27]:


# with open('./Q_var.pkl',"rb") as f:
#     Qtsq=pickle.load(f)
#     Qt_trans=pickle.load(f)
#     Qt_scat=pickle.load(f)
#     ell=pickle.load(f)


# # Load in Local Trispectrum

# # $\mathcal{C}_{\ell_1 \ell_2 \ell_3, \ell_4} \equiv 6 \int r^2 dr ~ \beta_{\ell_1}(r)\beta_{\ell_2}(r)\beta_{\ell_3}(r) \alpha_{\ell_4}(r)$

# In[47]:



        


# In[48]:

if rank==0:
    xeind=find_nearest(therm['x_e'],0.1)
    tau_rec=eta_class[xeind]
    delr1=50
    delr2=5
    indswitch=np.zeros([2],dtype=np.int32)
    rarr=np.arange(0,eta_0-tau_rec,delr1)
    indswitch[0]=rarr.shape[0]-1
    rarr=np.append(rarr,np.arange(rarr[-1],eta_0,delr2))
    indswitch[1]=rarr.shape[0]-1
    rarr=np.append(rarr,np.arange(rarr[-1],eta_0+2000,delr1))
    rarr=np.unique(rarr)
    interparr=kint[np.newaxis,:]*rarr[:,np.newaxis]
    l_mark=1
    jl=spherical_jn(1,interparr)
    jl_n=spherical_jn(0,interparr)
    alph,beta=gen_alph(T_l_0_2d,ell,kint,rarr,interparr)


# In[49]:



# In[50]:
if rank == 0:
    with open('C_ell_prog.txt','w') as f:
        print(('computing cal_C_ell'),file=f)


    print('computing cal_C_ell')
    retval=np.zeros([ell.size,ell.size,ell.size,ell.size])

    C_ell=np.asarray(gen_C_ellx4_slow(alph, beta, ell.astype(np.int32), rarr, delr1, delr2, indswitch, retval))*6#*(4*np.pi)**4
    with open('./C_ell_full_hi_times2.pkl',"wb") as f:
        pickle.dump(C_ell,f)
        pickle.dump(ell,f)


    # In[51]:


    del alph, beta


# In[28]:


# with open('./C_ell_full.pkl',"rb") as f:
#     C_ell=pickle.load(f)
#     ell=pickle.load(f)




# In[52]:


# varlist=np.asarray([10.77e-6,6.4e-6,12.48e-6]) #kelvin
# FWHMlist=np.asarray([9.66,7.27,5.01])*np.pi/(60*180.)
# #varlist=np.asarray([22e-6,29e-6,49e-6]) #kelvin #wmap??
# #FWHMlist=np.asarray([28,21,13])*np.pi/(60*180.)
# N_ell=comp_N_ell(varlist,FWHMlist,ell)
# Ctilde=C_l_comp_2d+N_ell


# # In[53]:


# C_sym=(C_ell+np.swapaxes(C_ell,0,3)+np.swapaxes(C_ell,2,3)+np.swapaxes(C_ell,1,3))
# denom=(Ctilde[:,np.newaxis,np.newaxis,np.newaxis]*Ctilde[np.newaxis,:,np.newaxis,np.newaxis]*Ctilde[np.newaxis,np.newaxis,:,np.newaxis]*Ctilde[np.newaxis,np.newaxis,np.newaxis,:])

# quad=C_sym**2*Qsq/denom


# # In[31]:


# F_PNG=trap_int(trap_int(trap_int(trap_int(quad,ell),ell),ell),ell)/(4*3*2)

# with open('results.txt','w') as f:
#     print(('F_png local: %s' % (1/np.sqrt(.779*F_PNG)/1e5)),file=f)

# print('F_png local:')
# print((1/np.sqrt(.779*F_PNG)/1e5))

# F_PNG_at_ell=trap_int(trap_int(trap_int(quad,ell),ell),ell)/(4*3*2)
# plt.rcParams['figure.figsize']=[12,8]

# fig,ax=plt.subplots(1,1)
# ax.plot(ell,F_PNG_at_ell*ell)
# ax.set_xscale('log')
# plt.savefig('F_PNG_ell.pdf')





# # # Compute Local Bias due to PBH

# # # $\langle \Delta g_{\rm NL}\rangle_{\rm pbh}= \frac{f_{\rm pbh}}{F^{\rm {PNG}}} \sum_{\ell_1 \leq \ell_2 \leq \ell_3 \leq \ell_4} \frac{\left(\mathcal{C}_{\ell_1 \ell_2 \ell_3, \ell_4} + 3 ~\textrm{perm} \right)}{C_{\ell_1} C_{\ell_2} C_{\ell_3} C_{\ell_4}}\left[\mathcal{A}_{(\ell_1 \ell_2 \ell_3 \ell_4)} (\mathcal{Q}^2)_{\ell_1 \ell_2 \ell_3 \ell_4} + \left(\mathcal{B}_{\ell_1 \ell_2, \ell_{34}} (\mathcal{Q \widetilde{Q}})_{\ell_1 \ell_2, \ell_3 \ell_4} + 5~ \textrm{perm} \right) \right] $

# # In[55]:


# sym_A=(cal_A+np.moveaxis(cal_A,(0,1,2),(2,0,1))+np.moveaxis(cal_A,(0,1,2),(1,2,0)))[:,:,:,np.newaxis]
# sym_A=(sym_A+np.moveaxis(sym_A,(0,1,2,3),(1,2,3,0))+np.moveaxis(sym_A,(0,1,2,3),(2,3,0,1))+np.moveaxis(sym_A,(0,1,2,3),(3,0,1,2)))
# add_B=(cal_B[:,:,:,np.newaxis]+cal_B[:,:,np.newaxis,:])
# tmp=(add_B)*QQt
# T_PBH=sym_A*Qsq+tmp+np.moveaxis(tmp,(0,1,2,3),(0,2,1,3))+np.moveaxis(tmp,(0,1,2,3),(0,2,3,1))+np.moveaxis(tmp,(0,1,2,3),(2,0,1,3))+np.moveaxis(tmp,(0,1,2,3),(2,0,3,1))+np.moveaxis(tmp,(0,1,2,3),(2,3,0,1))


# # In[56]:


# del cal_B, cal_A


# # In[71]:

# g_NL_quad=C_sym/denom*T_PBH/F_PNG/(4*3*2)
# g_NL=trap_int(trap_int(trap_int(trap_int(g_NL_quad,ell),ell),ell),ell)

# with open('results.txt','a') as f:
#     f.write('\n')
#     f.write('g_bias: %s' % g_NL)


# print('g bias:')
# print(g_NL)



# g_NL_at_ell=trap_int(trap_int(trap_int(g_NL_quad,ell),ell),ell)
# plt.rcParams['figure.figsize']=[12,8]

# fig,ax=plt.subplots(1,1)
# ax.plot(ell,g_NL_at_ell*ell)
# ax.set_xscale('log')
# plt.savefig('g_bias_ell.pdf')
# del g_NL_quad


# # # Compute f_pbh var

# # # $F^{\rm pbh} = \textrm{var}^{-1}(f_{\rm pbh}) = \sum_{\ell_1 \leq \ell_2 \leq \ell_3 \leq \ell_4} \frac1{C_{\ell_1}C_{\ell_2} C_{\ell_3} C_{\ell_4}} \Big{[}\mathcal{A}_{(\ell_1 \ell_2 \ell_3 \ell_4)}^2 (\mathcal{Q}^2)_{\ell_1 \ell_2 \ell_3 \ell_4}  + \mathcal{A}_{(\ell_1 \ell_2 \ell_3 \ell_4)}\left(\mathcal{B}_{\ell_1 \ell_2, \ell_{34}} (\mathcal{Q \widetilde{Q}})_{\ell_1 \ell_2, \ell_3 \ell_4} + 5~ \textrm{perm} \right)\nonumber\\
# #  ~~~~~~~~~~~~~~~~~~~ + \left((\mathcal{B}_{\ell_1 \ell_2, \ell_{34}})^2 (\mathcal{\widetilde{Q}}^2)_{\ell_1 \ell_2, \ell_3 \ell_4} + 5~ \textrm{perm} \right) + 2~\left(\mathcal{B}_{\ell_1 \ell_2, \ell_{34}}\mathcal{B}_{\ell_3 \ell_4, \ell_{12}} (\mathcal{\widetilde{Q} \widetilde{Q}}^{\rm T})_{\ell_1 \ell_2, \ell_3 \ell_4} + 2 \ \textrm{perm} \right)\nonumber\\
# #  ~~~~~~~~~~~~~~~~~~~~ + 2\left(\mathcal{B}_{\ell_1 \ell_2, \ell_{34}}\mathcal{B}_{\ell_1 \ell_3, \ell_{24}}(\mathcal{\widetilde{Q} \widetilde{Q}}^{\rm S})_{\ell_1, \ell_2 \ell_3, \ell_4} +11 ~\textrm{perm} \right)\Big{]}.$

# # In[ ]:


# tmp=(add_B)*QQt
# F_PBH=sym_A**2*Qsq+sym_A*(tmp+np.moveaxis(tmp,(0,1,2,3),(0,2,1,3))+np.moveaxis(tmp,(0,1,2,3),(0,3,1,2))+np.moveaxis(tmp,(0,1,2,3),(1,2,0,3))+np.moveaxis(tmp,(0,1,2,3),(1,3,0,2))+np.moveaxis(tmp,(0,1,2,3),(2,3,0,1)))
# tmp=add_B**2*Qtsq
# F_PBH+=tmp+np.moveaxis(tmp,(0,1,2,3),(0,2,1,3))+np.moveaxis(tmp,(0,1,2,3),(0,3,1,2))+np.moveaxis(tmp,(0,1,2,3),(1,2,0,3))+np.moveaxis(tmp,(0,1,2,3),(1,3,0,2))+np.moveaxis(tmp,(0,1,2,3),(2,3,0,1))
# tmp=2*add_B*np.moveaxis(add_B,(0,1,2,3),(2,3,0,1))*Qt_trans
# F_PBH+=tmp+np.moveaxis(tmp,(0,1,2,3),(0,3,1,2))+np.moveaxis(tmp,(0,1,2,3),(0,2,1,3))
# tmp=2*add_B*np.moveaxis(add_B,(0,1,2,3),(0,2,1,3))*Qt_scat
# F_PBH+=tmp+np.moveaxis(tmp,(0,1,2,3),(1,2,3,0))+np.moveaxis(tmp,(0,1,2,3),(2,3,0,1))+np.moveaxis(tmp,(0,1,2,3),(3,0,1,2))+np.moveaxis(tmp,(0,1,2,3),(3,1,2,0))+np.moveaxis(tmp,(0,1,2,3),(1,0,2,3))+np.moveaxis(tmp,(0,1,2,3),(1,3,0,2))+np.moveaxis(tmp,(0,1,2,3),(2,3,1,0))+np.moveaxis(tmp,(0,1,2,3),(3,0,2,1))+np.moveaxis(tmp,(0,1,2,3),(2,1,0,3))+np.moveaxis(tmp,(0,1,2,3),(0,1,3,2))+np.moveaxis(tmp,(0,1,2,3),(0,3,2,1))



# F_PBH_quad=F_PBH/denom/(4*3*2)
# F_PBH_dev=1/np.sqrt(trap_int(trap_int(trap_int(trap_int(F_PBH_quad,ell),ell),ell),ell))

# with open('results.txt','a') as f:
#     f.write('\n')
#     f.write('F_pbh sigma: %s' % F_PBH_dev)



# print('F_pbh sigma:')
# print(F_PBH_dev)
# del F_PBH_quad


# # In[ ]:


# F_PBH_dev_at_ell=(trap_int(trap_int(trap_int(F_PBH_quad,ell),ell),ell))



# # In[ ]:


# fig,ax=plt.subplots(1,1)
# ax.plot(ell,F_PBH_dev_at_ell*ell,marker='o')
# ax.set_xscale('log')
# plt.savefig('f_pbh_var_ell.pdf')




