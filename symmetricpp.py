# J.K. Ryan    
# 15 June 2018 

# import necessary stuff for code
import numpy as np
import sympy as sym


from numpy import *

import math
from math import *

#  Post-processing functions
import getkernelcoeff
#  import evalkernel
import xintsum


from getkernelcoeff import getkernelcoeff
#  from evalkernel import evalkernel
from xintsum import xintsum

#######################################################################

def symmetricpp(p,ell,RS,zEval):

    evalPoints = len(zEval)
    evalPoints = int(evalPoints)


    # Kernel coefficients -- ensures 2*RS+1 moments through polynomial reproduction
    # These are the B-spline weights
    cgam = np.zeros((2*RS+1))
    cgam = getkernelcoeff(ell,RS)

    
    # Get quadrature points and weights in order to evaluate the post-processing integrals:  
    # Approximation is a polynomial of degree p, kernel is a
    # polynomial of degree ell-1  ==> we need p+ell-1 = 2gpts-1, where n is the number of points.
    # Hence, gpts=(p+ell)/2.  If p+ell is odd, we want the first integer >= (p+ell)/2, hence the
    # ceiling function.
    gpts = int(ceil(0.5*(p+ell)))
    z = np.zeros((gpts))
    w = np.zeros((gpts))
    # Gauss-Legendre (default interval is [-1, 1])
    z, w = np.polynomial.legendre.leggauss(gpts)

    # Post-processor support is (xbar - kernelsupp*dx, xbar + kernelsupp*dx)
    kernelsupp = np.float(RS+0.5*ell) 
    # Make the element counter and integer value
    kwide = int(ceil(kernelsupp))
    # Total number of elements in the support
    pwide = 2*kwide+1
    print('In Symmetricpp:\n')
    print('    kernelsupp = ',kernelsupp,'    kwide = ',kwide,'    pwide = ',pwide,'\n')

    # Need to account for the case where the B-spline breaks are not aligned with the evaluation point.
    # This occurs with ell is odd (odd B-spline order)
    if ell % 2 == 0:
        kres = np.float(0)
    else:
        kres = np.float(1.0)

    # symcc is the symmetric post-processing matrix
    symcc = np.zeros((pwide,p+1,evalPoints))

    for j in arange(evalPoints): # Loop over element evaluation points

        if kres !=0 and zEval[j] > 0: # locate kernel break.  Done based on where the
                                      # evaluation point is  with respect to cell center 
                                      # if ell is odd
            kres = np.float(-1.0)
        
        zetaEval = zEval[j]+kres # This is the location of the kernel break within the element
                                 # for a uniform grid
        
        for kk1 in arange(pwide):
            kk = kk1-kwide
            # Integral evaluation arrays
        
            if ell % 2 == 0: #B-spline order is even.  
                             #Kernel breaks are the shifted evaluation point

                # evaluation od the first integral
                # in_{-1}^zEval[j] K(0.5*(zEval[j]-x)-kk)P[m][x] dx, where kk is 
                # the current element with respect the element of the post-processing point 
                ahat = np.float(-1.0)
                bhat = np.float(zetaEval)

                xintsum1 = xintsum(ahat,bhat,p,kwide,pwide,kernelsupp,z,w,RS,ell,cgam,kk,zetaEval,zEval[j])

                # evaluation of the second integral (later need to scale by 2)
                # int_zEval[j]^1 K(0.5*(zEval[j]-x)-kk)P[m][x] dx, where 
                # kk is the current element with respect the element of 
                # the post-processing point 
                ahat = np.float(zetaEval)
                bhat = np.float(1.0)

                xintsum2 = xintsum(ahat,bhat,p,kwide,pwide,kernelsupp,z,w,RS,ell,cgam,kk,zetaEval,zEval[j])
            else:  #B-spline order is odd, kernel breaks depend on location of evaluation point 
                   #with respect to the cell center
                
                if zEval[j] != 0:
                    ahat = np.float(-1.0)
                    bhat = np.float(zetaEval)
                    xintsum1 = xintsum(ahat,bhat,p,kwide,pwide,kernelsupp,z,w,RS,ell,cgam,kk,zetaEval,zEval[j])

                    ahat = np.float(zetaEval)
                    bhat = np.float(1.0)
                    xintsum2 = xintsum(ahat,bhat,p,kwide,pwide,kernelsupp,z,w,RS,ell,cgam,kk,zetaEval,zEval[j])

                else:
                    ahat = np.float(-1.0)
                    bhat = np.float(1.0)
                    xintsum1 = xintsum(ahat,bhat,p,kwide,pwide,kernelsupp,z,w,RS,ell,cgam,kk,zetaEval,zEval[j])
                    xintsum2 = np.zeros((p+1))

            # form the post-processing matrix = 0.5*(I1+I2)
            for m in arange(p+1):
                symcc[kk1][m][j] = 0.5*(xintsum1[m]+xintsum2[m])

    return symcc


########################################################################  
