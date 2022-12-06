# J.K. Ryan                                                                                                    # 15 June 2018 

# import necessary stuff for code                                                                                                                                   
import numpy as np
import sympy as sym


from numpy import *
from scipy import *
from scipy import integrate
from scipy.special import binom
#  import matplotlib.pyplot as plt
import math
#import np.linalg                                                                                                                                                   
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                                
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve

from math import *

#######################################################################  

# Obtain the B-spline weights that give the kernel coefficients.  
# This is done through polynomial reproduction:
# int_R K(x-y)y^m dy = x^m, m=0..,2*RS.
# If the B-spline order is large, this matrix become ill-conditioned.

def getkernelcoeff(ell,RS):
    # Define matrix to determine kernel coefficients
        A=np.zeros((2*RS+1,2*RS+1))
        for m in arange(2*RS+1):
            for gam in arange(2*RS+1):
                component = 0.
                for n in arange(m+1):
                    jsum = 0.
                    jsum = sum((-1)**(j+ell-1)*binom(ell-1,j)*((j-0.5*(ell-2))**(ell+n)-(j-0.5*ell)**(ell+n)) for j in arange(ell))
                    component += binom(m,n)*(gam-RS)**(m-n)*factorial(n)/factorial(n+ell)*jsum

                    A[m][gam] = component

        print('\n')
        print('Matrix for SIAC coefficients')
        print(A)
        print('\n')

        b=np.zeros((2*RS+1))
        b[0]=1.

        c = np.zeros((2*RS+1))
        #call the lu_factor function LU = linalg.lu_factor(A)
        Piv = scipy.linalg.lu_factor(A)
        #P, L, U = scipy.linalg.lu(A)
        #solve given LU and B
        c = scipy.linalg.lu_solve(Piv, b)


        print('SIAC coefficients:',c)

        # check coefficients add to one
        sumcoeff = sum(c[n] for n in arange(2*RS+1))
        print('Sum of coefficients',sumcoeff)


        return c


########################################################################  
