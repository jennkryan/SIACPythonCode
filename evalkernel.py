# J.K. Ryan
# 15 June 2018

# import necessary stuff for code                                                                                                                                   
import numpy as np
import sympy as sym


from numpy import *
from scipy import *
from scipy import integrate
from scipy.special import binom
import matplotlib.pyplot as plt
import math
#import np.linalg                                                                                                                                                   
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                                
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve

from math import *

import bspline


#######################################################################                                                                                            
#  evaluate the kernel:
#  K(x) = sum_gam c_gam psi^{ell}(x-gam),
#  where psi^{ell}(x-gam) is a B-spline of order ell centered at gam. 

def evalkernel(ell,RS,cgam,gpts,kerzeta):

    # Define B-spline breaks for a B-spline of order ell                                                                                                            
    bsbrks = np.linspace(-0.5*(2*RS+ell),0.5*(2*RS+ell),2*RS+ell+1)
    basis = bspline.Bspline(bsbrks,ell-1)

#    basis.plot()
#    bsbrks = np.zeros(ell+1)
#    for i in arange(ell+1):
#        bsbrks[i] = -0.5*ell+i


    fker = np.zeros((gpts))

    g = np.zeros((gpts, 2*RS+1))

    for n in arange(gpts): # summing over zetas
        g[n][:] = basis(kerzeta[n])*cgam[:]
    
        fker[n] = sum(g[n][jj] for jj in arange(2*RS+1))
                
    return fker

######################################################################## 
