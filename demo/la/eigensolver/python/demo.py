"This simple program illustrates the use of the SLEPc eigenvalue solver."

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2008-12-12"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *
import numpy

# Test for PETSc and SLEPc
try:
    PETScMatrix()
except:
    print "DOLFIN has not been configured with PETSc. Exiting."
    exit()
try:
    SLEPcEigenSolver()
except:
    print "DOLFIN has not been configured with SLEPc. Exiting."
    exit()

# Make sure we use the PETSc backend
dolfin_set("linear algebra backend", "PETSc")

# Build stiftness matrix
mesh = UnitSquare(64, 64)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)
A = PETScMatrix()
assemble(dot(grad(v), grad(u))*dx, tensor=A)

# Compute the first n eigenvalues
n = 10
esolver = SLEPcEigenSolver()
esolver.set("eigenvalue spectrum", "smallest magnitude")
esolver.solve(A, n)

print "Eigenvalue solver converged in %d iterations" % (esolver.getIterationNumber(),)

# Display eigenvalues
for i in range(n):
    (lr, lc) = esolver.getEigenvalue(i)
    print "Eigenvalue " + str(i) + ": " + str(lr)
