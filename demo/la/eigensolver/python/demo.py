# This simple program illustrates the use of the PETScEigenvalueSolver

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import numpy

A = dolfin.PETScMatrix(2,2)
A[0, 0] = 4.0
A[0, 1] = 1.0
A[1, 0] = 3.0
A[1, 1] = 2.0

A.apply()
print ""
print "Matrix A:"
A.disp()
print ""

B = dolfin.PETScMatrix(2,2)
B[0,0] = 4.0
B[0,1] = 0.0
B[1,0] = 0.0
B[1,1] = 1.0
B.apply()
print ""
print "Matrix B:"
B.disp()
print ""

# Create eigensolver of type LAPACK
esolver = SLEPcEigenvalueSolver(SLEPcEigenvalueSolver.lapack)

# Compute all eigenpairs of the generalised problem Ax = \lambda Bx
esolver.solve(A, B)

# Real and imaginary parts of an eigenvector
rr = dolfin.PETScVector(2)
cc = dolfin.PETScVector(2)

# Get the first eigenpair from the solver
emode = 0
err, ecc  = esolver.getEigenpair(rr, cc, emode)

# Display result
print ""
print "Eigenvalue, mode:\n", emode
print "real:\n", err
print "imag:\n", ecc
print "Eigenvalue vectors:"
print "real part:\n"
rr.disp()
print "\ncomplex part:\n"
cc.disp()

