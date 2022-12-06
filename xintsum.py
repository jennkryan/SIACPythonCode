# J.K. Ryan                                                                                                     
# 15 June 2018 
# import necessary stuff for code                                                                                                     
import numpy as np
import sympy as sym


from numpy import *
import math

from math import *

#  Post-processing functions                                                                                                          
import evalkernel
from evalkernel import evalkernel


####################################################################### 

# Evaluate the post-processing integrals using Gauss-Legendre quadrature.  The integral is:
# int_a^b K(0.5(zEval - x) - kk)P^(m)(x) dx,
# where K is the SIAC kernel using 2*RS+1 Bsplines of order ell, 
# zEval is the evaluation point, and P^(m) is the Legendre polynomial of degree m.

def xintsum(ahat,bhat,p,kwide,pwide,kernelsupp,z,w,RS,ell,cgam,kk,zetaEval,zEvalj):

    gpts = int(len(z))    
    xintsum = np.zeros((p+1))

    # Ensure integration does not go beyond the support of the kernel.
    intlow = np.float(zEvalj-2*kernelsupp-2.0*kk)
    intup = np.float(zEvalj+2*kernelsupp-2.0*kk)
    if ahat < intlow:
        ahat = intlow
    if bhat > intup:

        bhat = intup

    if ahat < bhat: # only perform the integration if ahat < bhat
                    # scale the integration interval to (-1,1) to use Gauss-Legendre quadrature
        abplus = 0.5*(ahat+bhat)
        abminus = 0.5*(bhat-ahat)
        zeta = np.zeros((gpts))
        zeta[:] = abminus*z[:]+abplus # quadrature coordinate

        # Evaluation coordinate for the kernel integration
        kerzeta = np.zeros((gpts))
        kerzeta[:] = 0.5*(zEvalj-zeta[:])-np.float(kk)

        # Obtain the kernel value at the gauss points
        fker = np.zeros((gpts))
        fker = evalkernel(ell,RS,cgam,gpts,kerzeta)

        # Legendre polynomials evaluated at the gauss points
        PLeg = np.zeros((p+1,gpts))
        for m in arange(p+1):
            if m==0:
                PLeg[m][:] = np.ones((gpts))
            elif m ==1:
                PLeg[m][:] = zeta
            else:
                PLeg[m][:] = (2.0*m-1.0)/np.float(m)*zeta[:]*PLeg[m-1][:]-(m-1.0)/np.float(m)*PLeg[m-2][:]
            
        # Obtain the integral value
        for m in arange(p+1):
            integralval = sum(fker[n]*PLeg[m][n]*w[n] for n in arange(gpts))
            xintsum[m] = abminus*integralval
        
    return xintsum
