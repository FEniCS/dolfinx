"""
This program illustrates the use of the SLEPc eigenvalue solver for
both standard and generalized eigenvalue problems."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2009-10-09"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.
# Modified by Marie Rognes, 2009.

from dolfin import *
import numpy

# Test for PETSc and SLEPc
if not has_la_backend("PETSc"):
    print "DOLFIN has not been configured with PETSc. Exiting."
    exit()

if not has_slepc():
    print "DOLFIN has not been configured with SLEPc. Exiting."
    exit()

# Make sure we use the PETSc backend
parameters["linear_algebra_backend"] = "PETSc"

# Build stiftness matrix
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)
A = PETScMatrix()
assemble(dot(grad(v), grad(u))*dx, tensor=A)

# Build mass matrix
M = PETScMatrix()
assemble(v*u*dx, tensor=M)

# Compute the n largest eigenvalues of A x = \lambda x
n = 3
esolver = SLEPcEigenSolver()
esolver.solve(A, n)

# Display eigenvalues
for i in range(n):
    lr, lc, r_vec, c_vec = esolver.get_eigenpair(i)
    print "Eigenvalue " + str(i) + ": " + str(lr)
    print "Eigenvector " + str(i) + ": " + str(r_vec.array())

# Compute the eigenvalues close to 10 of A x = \lambda M x
print
print "Computing eigenvalues of generalized problem"
esolver = SLEPcEigenSolver()
esolver.parameters["solver"] = "krylov-schur"
esolver.parameters["spectral_transform"] = "shift-and-invert"
esolver.parameters["spectral_shift"] = 10.0
esolver.solve(A, M, n)

m = esolver.get_number_converged()
print "Number of converged eigenvalues: %d" % m
for i in range(m):
    r, c = esolver.get_eigenvalue(i)
    print "Real(Eigenvalue) " + str(i) + ": " + str(r)


# Test a different eigensolver type
print
print "Testing lanczos eigenvalue solver (not converging)"
esolver = SLEPcEigenSolver()
esolver.parameters["problem_type"] = "hermitian"
esolver.parameters["solver"] = "lanczos"
esolver.solve(M)

