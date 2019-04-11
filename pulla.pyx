from libc.stdlib cimport srand, rand, RAND_MAX

cdef extern from 'math.h':
    double log(double x)

def pulla(double astep, double Eistep, double zfinal, double ptmpco, double me, double sigT,int seed):
    cdef double dlna, ptmp, ppull, pxstep
    ptmp=ptmpco/astep**(3/2.)*(3/8.)*sigT*me/(Eistep*Eistep)*(2*(Eistep**3 + 9*Eistep*Eistep*me+8*Eistep*me*me+2*me**3)/((2*Eistep+me)*(2*Eistep+me))+(Eistep-2*me-2*me*me/Eistep)*log(1+2*Eistep/me))
    dlna=min(0.001/ptmp,0.001)
    srand(seed)
    ppull = float(rand())/(float(RAND_MAX))
    pxstep=ptmp*dlna
    while pxstep < ppull:
        ppull = float(rand())/(float(RAND_MAX))
        Eistep=Eistep*astep/(astep*(1+dlna))
        astep=astep*(1+dlna)
        if 1/astep-1 < zfinal:
            break
        ptmp=ptmpco/astep**(3/2.)*(3/8.)*sigT*me/(Eistep*Eistep)*(2*(Eistep**3 + 9*Eistep*Eistep*me+8*Eistep*me*me+2*me**3)/((2*Eistep+me)*(2*Eistep+me))+(Eistep-2*me-2*me*me/Eistep)*log(1+2*Eistep/me))
        dlna=min(0.001/ptmp,0.001)
        pxstep=ptmp*dlna
    return astep, Eistep
        
