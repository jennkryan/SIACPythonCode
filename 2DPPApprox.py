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

import symmetricpp

from symmetricpp import symmetricpp


########################################################################


#  MAIN PART OF PROGRAM
# Set number of evaluation points per element
print('To see the pattern of oscillations in the approximation, plot at many points per element.')
evalPoints = input('Input number of evaluation points per element (0 for nodes):  ');
evalPoints = int(evalPoints)

# Define number of elements, the polynomial degree
print('Define the parameters for the mesh')
Nx = input('Input number of elements in x:  ');
Ny = input('Input number of elements in y:  ');
print('Define the order of approximation (p+1), where p = degree.')
p = input('Input polynomial degree (NOT ORDER):  ');   #polynomial degree
Nx = int(Nx)
Ny = int(Ny)
p = int(p)

if evalPoints == 0:
    evalPoints = p + 1

# Get quadrature points and weights

z = np.zeros((p+1))
w = np.zeros((p+1))
# Gauss-Legendre (default interval is [-1, 1])
z, w = np.polynomial.legendre.leggauss(p+1)

# Now  get quadrature points and weights for the evaluation points
zEval = np.zeros((evalPoints))
wEval = np.zeros((evalPoints))
zEval, wEval = np.polynomial.legendre.leggauss(evalPoints)

# Evaluate the Legendre polynomials at p+1 points per element
LegPolyAtz = np.zeros((p+1,p+1))
LegMass = np.zeros((p+1))
for i in range(p+1):
    if i==0:
        LegPolyAtz[i][:] = 1.0
    elif i ==1:
        LegPolyAtz[i][:] = z
    else:
        LegPolyAtz[i][:] = (2*i-1)/i*z[:]*LegPolyAtz[i-1][:]-(i-1)/i*LegPolyAtz[i-2][:]

    LegMass[i] = np.dot(w,LegPolyAtz[i][:]*LegPolyAtz[i][:])

#Evaluate the Legendre polynomials at the evaluation points
print('Using Legendre polynomials for basis.  To convert to nodal, use Vandermonde matrix.')
LegPolyAtzEval = np.zeros((p+1,evalPoints))
for i in range(p+1):
    if i==0:
        LegPolyAtzEval[i][:] = 1.0
    elif i ==1:
        LegPolyAtzEval[i][:] = zEval
    else:
        LegPolyAtzEval[i][:] = (2*i-1)/i*zEval[:]*LegPolyAtzEval[i-1][:]-(i-1)/i*LegPolyAtzEval[i-2][:]

# ASSUMING UNIFORM INTERVALS.  DOMAIN IS [0,1].
xmax = np.float_(1.0)
xmin = np.float_(0.0)
xlength = xmax - xmin
hx = xlength/Nx
x_grid = np.zeros((Nx+1))
for k in range(Nx+1):
    x_grid[k] = xmin + float(k)*hx

ymax = xmax
ymim = xmin
ylength = ymax - ymim
hy = ylength/Ny
y_grid = x_grid


# Define modes of approximation by performing an L2-projection onto the space of piecewise Legendre polynomials
uhat = np.zeros((Nx,Ny,p+1,p+1))
for nelx in range(Nx):
    term1 = np.zeros((p+1))
    term3 = np.zeros((p+1))
    for mx in range(p+1):
        for ix in range(p+1):
            xrg = 0.5*hx*(z[ix]+1.0) + x_grid[nelx]
            term1[mx] = term1[mx] + np.sin(2.0*math.pi*xrg)*LegPolyAtz[mx][ix]/LegMass[mx]*w[ix]
            term3[mx] = term3[mx] + np.cos(2.0*math.pi*xrg)*LegPolyAtz[mx][ix]/LegMass[mx]*w[ix]

    for nely in range(Ny):
        term2 = np.zeros((p+1))
        term4 = np.zeros((p+1))
        for my in range(p+1):
            for jy in range(p+1):
                yrg = 0.5*hy*(z[jy]+1.0) + y_grid[nely]
                term2[my] = term2[my] + np.cos(2.0*math.pi*yrg)*LegPolyAtz[my][jy]/LegMass[my]*w[jy]
                term4[my] = term4[my] + np.sin(2.0*math.pi*yrg)*LegPolyAtz[my][jy]/LegMass[my]*w[jy]


        for mx in range(p+1):
            for my in range(p+1):
                uhat[nelx][nely][mx][my] = term1[mx]*term2[my] + term3[mx]*term4[my]

# Gather values in order to plot the exact solution and the projection
xEval=[]
yEval=[]
fExact=[]
fApprox=[]

NxEval = Nx*evalPoints
NyEval = Ny*evalPoints


y1D=[]
Exactslice=[]
Approxslice=[]
for nelx in range(Nx):
    for ix in range(evalPoints):

        xrg = 0.5*hx*(zEval[ix]+1.0) + x_grid[nelx]

        for nely in range(Ny):
            for jy in range(evalPoints):


                yrg = 0.5*hy*(zEval[jy]+1.0) + y_grid[nely]
                xEval.append(xrg)
                yEval.append(yrg)

                fval = np.sin(2.0*math.pi*(xrg+yrg))
                fExact.append(fval)

                uval = np.float_(0.0)
                for mx in range(p+1):
                    for my in range(p+1):
                        uval = uval + uhat[nelx][nely][mx][my]*LegPolyAtzEval[mx][ix]*LegPolyAtzEval[my][jy]


                fApprox.append(uval)


ApproxErr = np.subtract(fExact, fApprox)
PtwiseAErr = np.abs(ApproxErr)
LinfAErr = np.max(PtwiseAErr)

#############################################################################################################

# Define kernel smoothness
print('Here we begin defining the parameters for the SIAC kernel.')
print('The higher the smoothness, the more oscillations are supressed (more diffusion).')
ellp2 = input('Input smoothness required (>= -1).  -1 = weak continuity, 0 = continuous:  ');
ellp2 = int(ellp2)
ell = ellp2 + 2

# Define the number of splines (2*RS+1)
print('The number of moments that the kernel will satisfy is RS+1, ensuring that we at least maintain the order of the scheme')
print('Hence,  RS = max(ceil(0.5*(p+ell-1)),ceil(0.5*p)), with p+1 = DG order and ell = smoothness + 2.')
RS = int(max(ceil(0.5*(p+ell-1)),ceil(0.5*p)));
kwide = ceil(RS+0.5*ell)

print('The scaling, LH, is typically taken to be the mesh size, but there are some cases where a larger scaling is desired.')
L=1

#symcc is the symmetric post-processing matrix
symcc = symmetricpp(p,ell,RS,zEval)

PPxEval=[]
PPyEval=[]
PPfExact=[]
PPfApprox=[]
onedcoord=[]
f1d=[]
pp1d=[]

for nelx in range(Nx):
    for ix in range(evalPoints):
        xrg = 0.5*hx*(zEval[ix]+1.0) + x_grid[nelx]
        PPxEval.append(xrg)
        for nely in range(Ny):
            for jy in range(evalPoints):
                yrg = 0.5*hy*(zEval[jy]+1.0) + y_grid[nely]

                f=np.sin(2.0*math.pi*(xrg+yrg))
                PPyEval.append(yrg)
                PPfExact.append(f)

                # set indices to form post-processed solution
                upost = 0.0
                if kwide <= nelx <= Nx-2-kwide:
                    for kkx in range(2*kwide+1):
                        kk2x = kkx - kwide
                        xindex = nelx + kk2x

                        # interior elements, use symmetric filter
                        if kwide <= nely <= Ny-2-kwide:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                yindex = nely + kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                        # boundary elements, either turn of boundary filtering or use this bit.
                        elif nely < kwide:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                if nely+kk2y<0:
                                    yindex = Ny+nely+kk2y
                                else:
                                    yindex = nely+kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]

                        else:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                if kk2y <=0:
                                    yindex = nely + kk2y
                                else:
                                    yindex = nely-Ny+kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                # boundary filtering, either turn of boundary filtering or use this bit.
                elif nelx < kwide:
                    for kkx in range(2*kwide+1):
                        kk2x = kkx - kwide
                        if nelx+kk2x <0:
                            xindex = Nx+nelx+kk2x
                        else:
                            xindex = nelx+kk2x
                        if kwide <=nely <= Ny-2-kwide:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                yindex = nely+kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                        elif nely < kwide:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                if nely + kk2y < 0:
                                    yindex = Ny + nely + kk2y
                                else:
                                    yindex = nely + kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                        else:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                if kk2y <=0:
                                    yindex = nely + kk2y
                                else:
                                    yindex = nely - Ny +kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                else:
                    for kkx in range(2*kwide+1):
                        kk2x = kkx - kwide
                        if kk2x <= 0:
                            xindex = nelx + kk2x
                        else:
                            xindex = nelx-Nx+kk2x
                        if kwide <=nely <= Ny-2-kwide:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                yindex = nely+kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                        elif nely < kwide:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                if nely + kk2y < 0:
                                    yindex = Ny + nely + kk2y
                                else:
                                    yindex = nely + kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]
                        else:
                            for kky in range(2*kwide+1):
                                kk2y = kky - kwide
                                if kk2y<=0:
                                    yindex = nely + kk2y
                                else:
                                    yindex = nely - Ny +kk2y
                                for mx in range(p+1):
                                    for my in range(p+1):
                                        upost = upost + symcc[kkx][mx][ix]*symcc[kky][my][jy]*uhat[xindex][yindex][mx][my]


                PPfApprox.append(upost)
#                if (nelx == 4) and (ix == 0):
#                    onedcoord.append(yrg)
#                    f1d.append(f)
#                    pp1d.append(upost)


#ppdiff = np.subtract(f1d,pp1d)
#ppSliceerr = np.abs(ppdiff)


PPApproxErr = np.subtract(PPfExact, PPfApprox)
PPPtwiseAErr = np.abs(PPApproxErr)
PPLinfAErr = np.max(PPPtwiseAErr)


print('\n')
print('Checking that we obtain higher order accuracy')
print('Nx = ',Nx,'    Ny = ',Ny,'    p = ',p)
print('L-inf Error for the Projection =',LinfAErr)
print('L-inf Error for the Post-processed Projection =',PPLinfAErr)
print('\n')
