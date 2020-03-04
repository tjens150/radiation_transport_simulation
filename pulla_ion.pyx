from libc.stdlib cimport srand, rand, RAND_MAX
from libc.stdio cimport printf
from libc.math cimport M_PI
from cpython cimport array

cdef extern from "math.h":
    double log(double x)

cdef extern from "math.h":
    double sqrt(double x)

cdef extern from "math.h":
    double exp(double x)

cdef extern from "math.h":
    double atan(double x)

cdef extern from "math.h":
    double floor(double x)

cdef extern from "math.h":
    double pow(double x, double y)

cdef double eta_H(double E):
    return 1./sqrt(E/13.6-1) #13.6 for ionizing Hydrogen

cdef double sig_H(double E):
    cdef double eta
    eta=eta_H(E)
    # 2**9*M_PI**2*rnot**2/(3*alph**3)*Eth**4=1.178E-15 m^2*eV
    return 1.178e-15*(1/E)**4*exp(-4*eta*atan(1/eta))/(1-exp(-2*M_PI*eta))


cdef double HubbleRate(double a,double H0, double omegaM,double omegaK,double omegaDE,double omegaR):
    return H0*sqrt(omegaM/(a*a*a)+omegaK/(a*a)+omegaDE+omegaR/(a*a*a*a))


cdef double sig_He(double E):
    return -12.*sig_H(E)+5.1e-24*pow(250./E,3.3)

cdef double rec_interp1d(double x0, double dx, double *ytab, int Nx, double x):

    cdef long ix
    cdef double frac


    #/* Identify location to interpolate */
    ix = <long>floor((x-x0)/dx)
    if (ix<1):
       ix=1 
    if (ix>Nx-3):
       ix=Nx-3
    frac = (x-x0)/dx-ix
    ytab += ix-1

    #/* Return value */
    return(
      -ytab[0]*frac*(1.-frac)*(2.-frac)/6.
      +ytab[1]*(1.+frac)*(1.-frac)*(2.-frac)/2.
      +ytab[2]*(1.+frac)*frac*(2.-frac)/2.
      -ytab[3]*(1.+frac)*frac*(1.-frac)/6.
    )


def pulla(double astep, double Eistep, double afinal, int Nxe, double ptmpco, double aeq, double me, double sigT,int seed,double nh0, double YHe, double H0, double omegaM, double omegaK, double omegaDE, double omegaR, double[:] xe):
    cdef double dlna, ppull, pxstep, ptmp, ppull_H, ppull_He, pxion, xei
    cdef int flag = 0
    cdef array.array xearr=array.array('d',xe)
    if Eistep < 2000.: #about 0.005*me
       ptmp=ptmpco/sqrt(astep*astep*astep*(1+aeq/astep))*sigT #thomson limit
    else:
       #ptmpco is c*nh0/sqrt(H0*omegaM)
       #probability to scatter in dlna, long expression comes from sigma
       ptmp=ptmpco/sqrt(astep*astep*astep*(1+aeq/astep))*(3/8.)*sigT*me/(Eistep*Eistep)*(2.*(Eistep**3 + 9.*Eistep*Eistep*me+8.*Eistep*me*me+2.*me**3)/((2.*Eistep+me)*(2.*Eistep+me))+(Eistep-2.*me-2.*me*me/Eistep)*log(1.+2.*Eistep/me))
    dlna=min(0.001/ptmp,0.001)
    srand(seed)
    ppull = float(rand())/(float(RAND_MAX))
    pxstep=ptmp*dlna
    while pxstep< ppull:
        pxion=nh0/(astep*astep*astep)*299792458.*dlna/HubbleRate(astep,H0,omegaM,omegaK,omegaDE,omegaR)
        ppull_H = float(rand())/(float(RAND_MAX))
        xei=rec_interp1d(1/afinal-1.-4.,1.,xearr.data.as_doubles,Nxe,1/astep-1.)
        #printf("%f, %f,%f, %d\n",xei,1/astep-1.,1/afinal-1.,Nxe)
        if ppull_H < pxion*(1.-xei)*sig_H(Eistep): #does not account for E<ionization
           flag=1
           break
        ppull_He = float(rand())/(float(RAND_MAX))
        if ppull_He < pxion*sig_He(Eistep)*(YHe/(1-YHe))/4.:
           flag=2
           break
        ppull=float(rand())/(float(RAND_MAX))
        Eistep=Eistep/(1+dlna)
        astep=astep*(1+dlna)
        if astep > afinal:
           break
        if Eistep < 2000.:
           ptmp=ptmpco/sqrt(astep*astep*astep*(1+aeq/astep))*sigT
        else:
           ptmp=ptmpco/sqrt(astep*astep*astep*(1+aeq/astep))*(3/8.)*sigT*me/(Eistep*Eistep)*(2.*(Eistep**3 + 9.*Eistep*Eistep*me+8.*Eistep*me*me+2.*me**3)/((2.*Eistep+me)*(2.*Eistep+me))+(Eistep-2.*me-2.*me*me/Eistep)*log(1.+2.*Eistep/me))
        dlna=min(0.001/ptmp,0.001)
        pxstep=ptmp*dlna
    return astep, Eistep, flag
