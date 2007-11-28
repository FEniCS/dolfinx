# This simple program illustrates the use of the PETScEigenvalueSolver

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import numpy

#
# THIS DEMO IS CURRENTLY NOT WORKING, SEE NOTE IN CODE.
#

# Set up two simple test matrices (2 x 2)
A_array = numpy.array([[4.0, 1.0], [3.0, 2.0]])
B_array = numpy.array([[4.0, 0.0], [0.0, 1.0]])

position = numpy.array([0, 1], 'uint')

A = PETScMatrix(2,2)
A.set(A_array, 2, position, 2, position)
A.apply()
print ""
print "Matrix A:"
A.disp()
print ""

B = PETScMatrix(2,2)
B.set(B_array, 2, position, 2, position)
B.apply()
print ""
print "Matrix B:"
B.disp()
print ""

# Create eigensolver of type LAPACK
esolver = SLEPcEigenvalueSolver(SLEPcEigenvalueSolver.lapack)

# Compute all eigenpairs of the generalised problem Ax = \lambda Bx
esolver.solve(A, B)

# Real and imaginary parts of an eigenvalue  
err = 0.0
ecc = 0.0

# Real and imaginary parts of an eigenvector
rr = PETScVector(2)
cc = PETScVector(2)

# Get the first eigenpair from the solver
emode = 0
esolver.getEigenpair(err, ecc, rr, cc, emode)

#ERROR:
# NotImplementedError: Wrong number of arguments for overloaded function 'SLEPcEigenvalueSolver_getEigenpair'.
#   Possible C/C++ prototypes are:
#     getEigenpair(dolfin::real &,dolfin::real &,dolfin::PETScVector &,dolfin::PETScVector &)
#     getEigenpair(dolfin::real &,dolfin::real &,dolfin::PETScVector &,dolfin::PETScVector &,int const)

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



