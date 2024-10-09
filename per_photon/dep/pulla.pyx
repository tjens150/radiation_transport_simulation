from libc.stdlib cimport srand, rand, RAND_MAX

cdef extern from "math.h":
    double log(double x)

cdef extern from "math.h":
    double sqrt(double x)

def pulla(double astep, double Eistep, double afinal, double ptmpco, double aeq, double me, double sigT,int seed):
    cdef double dlna, ppull, pxstep, ptmp
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
    return astep, Eistep
